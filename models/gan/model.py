#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from IPython import display
import tensorflow as tf
import soundfile as sf
import numpy as np
import matplotlib
import time
import PIL
import os

class GAN(object):
    """
       Class to represent GAN (SV) models with model saving / loading and playback & recording capabilities
    """

    def __init__(self, name='', id=-1, gender='neutral', latent_dim=100, slice_len=16384):
        """
        Method to initialize a gan model that will be saved in 'data/pt_models/{name}'
        :param name:        String id for this model
        :param id:          Version id for this model - default: auto-increment value along the folder 'data/pt_models/{name}'
        :param gender:      Gender against which this gan is optimized
        :param latent_dim:  Size of the input latent vector
        :param slice_len:   Number of samples of the generated audio
        """
        self.name = name
        self.gender = gender
        self.latent_dim = latent_dim
        self.slice_len = slice_len
        self.dir = os.path.join('.', 'data', 'pt_models', self.name, self.gender)
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        self.id = len(os.listdir(self.dir)) if id < 0 else id

    def save(self):
        """
        Method to save the weights of this model in 'data/pt_models/{name}/v{id}/model_weights.tf'
        """
        print('>', 'saving', self.name, 'model')
        if not os.path.exists(os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id)))):
            os.makedirs(os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id))))
        self.generator.save_weights(os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id)), 'model_generator_weights.tf'))
        self.discriminator.save_weights(os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id)), 'model_discriminator_weights.tf'))
        print('>', 'saved', self.name, 'generator model in', os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id)), 'model_generator_weights.tf'))
        print('>', 'saved', self.name, 'discriminator model in', os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id)), 'model_discriminator_weights.tf'))

    def load(self):
        """
        Method to load the weights of this model from 'data/pt_models/{name}/v{id}/model_weights.tf'
        """
        print('>', 'loading', self.name, 'model')
        if os.path.exists(os.path.join(self.dir)):
            if os.path.exists(os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id)))):
                self.generator.load_weights(os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id)), 'model_generator_weights.tf'))
                self.discriminator.load_weights(os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id)), 'model_discriminator_weights.tf'))
                print('>', 'loaded generator weights from', os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id)), 'model_generator_weights.tf'))
                print('>', 'loaded discriminator weights from', os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id)), 'model_discriminator_weights.tf'))
            else:
                print('>', 'no pre-trained generator weights from', os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id)), 'model_generator_weights.tf'))
                print('>', 'no pre-trained discriminator weights from', os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id)), 'model_discriminator_weights.tf'))
        else:
            print('>', 'no directory for', self.name, 'model at', os.path.join(self.dir))

    def get_generator(self):
        """
        Method to get the generator of this gan
        :return:    A generator
        """
        return self.generator

    def build_discriminator_model(self):
        """
        Method to build a discriminator
        """
        raise NotImplementedError

    def build_generator_model(self):
        """
        Method to build a generator
        """
        raise NotImplementedError

    def build(self):
        """
        Method to build a gan and eventually load its weights
        """
        self.generator = self.build_generator_model()
        self.discriminator = self.build_discriminator_model()
        self.load()

    def discriminator_loss(self, x, G_z, D_x, D_G_z):
        """
        Method to compute the discriminator loss
        :param real_output:     Real audio samples
        :param fake_output:     Fake audio samples
        :return:                Discriminator loss
        """
        disc_loss = tf.math.reduce_mean(D_G_z) - tf.math.reduce_mean(D_x)
        interpolated_shape = [len(x), 1, 1] if self.name == 'wavegan' else [len(x), 1, 1, 1]
        interpolates = x + tf.random.uniform(shape=interpolated_shape, minval=0., maxval=1.) * (G_z - x)
        with tf.GradientTape() as disc_tape:
            disc_tape.watch(interpolates)
            disc_interp = self.discriminator(interpolates)
        gradients = disc_tape.gradient(disc_interp, [interpolates])[0]
        slopes = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(gradients), axis=[1, 2]))
        disc_loss += 10 * tf.math.reduce_mean((slopes - 1.) ** 2.)
        return disc_loss

    def generator_loss(self, D_G_z):
        """
        Method to compute the generator loss
        :param D_G_z:     Fake audio samples
        :return:          Generator loss
        """
        return -tf.math.reduce_mean(D_G_z)

    @tf.function
    def train_step(self, x):
        """
        Method to perform one training step for this gan
        :param x:           Current batch data
        :return:            (generator loss, discriminator loss)
        """

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            z = tf.random.normal([len(x), self.latent_dim])

            G_z = self.generator(z)
            D_x = self.discriminator(x)
            D_G_z = self.discriminator(G_z)

            gen_loss = self.generator_loss(D_G_z)
            disc_loss = self.discriminator_loss(x, G_z, D_x, D_G_z)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return gen_loss, disc_loss

    def train(self, train_data, epochs, steps_per_epoch, batch):
        """
        Method to train a gan
        :param train_data:          Training data pipeline
        :param epochs:              Number of training epochs
        :param steps_per_epoch:     Number of steps per epoch
        :param batch:               Size of a training batch
        """

        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

        for epoch in range(epochs):
            tf.keras.backend.set_learning_phase(1)

            gen_losses = []
            disc_losses = []
            for step, batch_data in enumerate(train_data):
                t1 = time.time()
                gen_loss, disc_loss = self.train_step(batch_data)
                t2 = time.time()
                gen_losses.append(gen_loss.numpy())
                disc_losses.append(disc_loss.numpy())
                print('\r>', 'epoch', epoch+1, 'of', epochs, '| eta', str((t2-t1)*(steps_per_epoch-step)//60) + 'm', '| step', step+1, 'of', steps_per_epoch, '| gen_loss', round(np.mean(gen_losses), 5), '| disc_loss', round(np.mean(disc_losses), 5), end='')

            print()

            self.preview(self.generator)
            self.save()

    def preview(self, model, num_examples_to_generate=1):
        """
        Method to create audio samples of this gan as a preview
        :param model:                       Generator model
        :param num_examples_to_generate:    Number of audio samples to be generated
        """
        tf.keras.backend.set_learning_phase(0)
        display.clear_output(wait=True)
        predictions = model(tf.random.normal([num_examples_to_generate, self.latent_dim]), training=False)

        plt.figure(figsize=(32, 32))
        plt.plot(np.squeeze(predictions[0, :, :]).ravel())
        plt.axis('off')

        if not os.path.exists(os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id)))):
            os.makedirs(os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id))))

        plt.savefig(os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id)), 'preview.png'))
        sf.write(os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id)), 'preview.wav'), np.squeeze(predictions[0, :, :]), 16000)

        plt.close()

        print('> saved preview in', os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id)), 'preview.wav'))

