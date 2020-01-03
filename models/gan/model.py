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

    def __init__(self, name='', id=-1, gender='neutral', latent_dim=100, slice_len=16384):
        self.name = name
        self.gender = gender
        self.latent_dim = latent_dim
        self.slice_len = slice_len
        self.dir = os.path.join('.', 'data', 'pt_models', self.name, self.gender)
        self.id = str(len(os.listdir(self.dir))) if id < 0 else id

    def save(self):
        print('>', 'saving', self.name, 'model')
        if not os.path.exists(os.path.join(self.dir, 'v' + str(self.id))):
            os.makedirs(os.path.join(self.dir, 'v' + str(self.id)))
        self.generator.save_weights(os.path.join(self.dir, 'v' + str(self.id), 'model_generator_weights.tf'))
        self.discriminator.save_weights(os.path.join(self.dir, 'v' + str(self.id), 'model_discriminator_weights.tf'))
        print('>', 'saved', self.name, 'generator model in', os.path.join(self.dir, 'v' + str(self.id), 'model_generator_weights.tf'))
        print('>', 'saved', self.name, 'discriminator model in', os.path.join(self.dir, 'v' + str(self.id), 'model_discriminator_weights.tf'))

    def load(self):
        print('>', 'loading', self.name, 'model')
        if os.path.exists(os.path.join(self.dir)):
            if os.path.exists(os.path.join(self.dir, 'v' + str(self.id))):
                self.generator.load_weights(os.path.join(self.dir, 'v' + str(self.id), 'model_generator_weights.tf'))
                self.discriminator.load_weights(os.path.join(self.dir, 'v' + str(self.id), 'model_discriminator_weights.tf'))
                print('>', 'loaded generator weights from', os.path.join(self.dir, 'v' + str(self.id), 'model_generator_weights.tf'))
                print('>', 'loaded discriminator weights from', os.path.join(self.dir, 'v' + str(self.id), 'model_discriminator_weights.tf'))
            else:
                print('>', 'no pre-trained generator weights from', os.path.join(self.dir, 'v' + str(self.id), 'model_generator_weights.tf'))
                print('>', 'no pre-trained discriminator weights from', os.path.join(self.dir, 'v' + str(self.id), 'model_discriminator_weights.tf'))
        else:
            print('>', 'no directory for', self.name, 'model at', os.path.join(self.dir))

    def get_generator(self):
        return self.generator

    def build_discriminator_model(self, training=True):
        raise NotImplementedError

    def build_generator_model(self, training=True):
        raise NotImplementedError

    def build(self):
        self.generator = self.build_generator_model()
        self.discriminator = self.build_discriminator_model()
        self.load()

    def discriminator_loss(self, real_output, fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        return cross_entropy(tf.ones_like(fake_output), fake_output)

    @tf.function
    def train_step(self, batch, batch_data):

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(tf.random.normal([batch, self.latent_dim]), training=True)

            real_output = self.discriminator(batch_data, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return gen_loss, disc_loss

    def train(self, train_data, epochs, steps_per_epoch, batch):
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

        for epoch in range(epochs):
            print('Epoch', epoch+1, 'of', epochs)
            tf.keras.backend.set_learning_phase(1)

            step = 0
            for batch_data in train_data:
                gen_loss, disc_loss = self.train_step(batch, batch_data)
                print('> Step', step + 1, 'of', steps_per_epoch, '\tgen_loss', gen_loss.numpy(), '\tdisc_loss', disc_loss.numpy())
                step += 1

            self.preview(self.generator)

            self.save()

    def preview(self, model, num_examples_to_generate=1):
        tf.keras.backend.set_learning_phase(0)
        display.clear_output(wait=True)
        predictions = model(tf.random.normal([num_examples_to_generate, self.latent_dim]), training=False)

        plt.figure(figsize=(32, 32))
        plt.plot(np.squeeze(predictions[0, :, :]).ravel())
        plt.axis('off')

        if not os.path.exists(os.path.join(self.dir, 'v' + str(self.id))):
            os.makedirs(os.path.join(self.dir, 'v' + str(self.id)))

        plt.savefig(os.path.join(self.dir, 'v' + str(self.id), 'fake.png'))
        sf.write(os.path.join(self.dir, 'v' + str(self.id), 'fake.wav'), np.squeeze(predictions[0, :, :]), 16000)

        plt.close()

