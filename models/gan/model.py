#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from IPython import display
import tensorflow as tf
import soundfile as sf
import numpy as np
from tqdm import tqdm
from helpers import plotting
from itertools import product
import matplotlib
import time
import os

class GAN(object):
    """
       Class to represent GAN (SV) models with model saving / loading and playback & recording capabilities
    """

    def __init__(self, name='', id=-1, gender='neutral', latent_dim=100, slice_len=16384, lr=1e-4):
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
        self.performance = {'gen': [], 'disc': [], 'accuracy': []}
        self.prediction_history = {'real': [], 'fake': []}
        self.dir = os.path.join('.', 'data', 'pt_models', self.name, self.gender)
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        self.id = len(os.listdir(self.dir)) if id < 0 else id
        
        self.generator_optimizer = tf.keras.optimizers.Adam(lr)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(lr)
        
    @property
    def version(self):
        return f'v{self.id:03d}'

    def save(self):
        """
        Method to save the weights of this model in 'data/pt_models/{name}/v{id}/model_weights.tf'
        """
#         print('>', 'saving', self.name, 'model')
        
        if not os.path.exists(os.path.join(self.dir, self.version)):
            os.makedirs(os.path.join(self.dir, self.version))
        
        self.generator.save_weights(os.path.join(self.dir, self.version, 'gen.h5'))
        self.discriminator.save_weights(os.path.join(self.dir, self.version, 'disc.h5'))
#         print('>', 'saved', self.name, 'generator model in', os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id)), 'model_generator_weights.tf'))
#         print('>', 'saved', self.name, 'discriminator model in', os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id)), 'model_discriminator_weights.tf'))

    def load(self):
        """
        Method to load the weights of this model from 'data/pt_models/{name}/v{id}/model_weights.tf'
        """
        print('>', 'loading', self.name, 'model')
        if os.path.exists(os.path.join(self.dir, self.version)):
            if len(os.listdir(os.path.join(self.dir, self.version))) > 0:
                self.generator.load_weights(os.path.join(self.dir, self.version, 'gen.h5'))
                self.discriminator.load_weights(os.path.join(self.dir, self.version, 'disc.h5'))
                print('>', 'loaded generator from', os.path.join(self.dir, self.version, 'gen'))
                print('>', 'loaded discriminator from', os.path.join(self.dir, self.version, 'disc'))
            else:
                print('>', 'no pre-trained generator from', os.path.join(self.dir, self.version, 'gen'))
                print('>', 'no pre-trained discriminator from', os.path.join(self.dir, self.version, 'disc'))
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

    def discriminator_loss(self, x, G_z, D_x, D_G_z, gradient_penalty=True):
        """
        Method to compute the discriminator loss with gradient norm penalty [1].
        
        References:
        [1] https://papers.nips.cc/paper/7159-improved-training-of-wasserstein-gans.pdf
        
        :param real_output:     Real audio samples
        :param fake_output:     Fake audio samples
        :return:                Discriminator loss
        """
        
        # The original GAN loss
        disc_loss =  tf.math.reduce_mean(D_G_z) - tf.math.reduce_mean(D_x)
        
        if not gradient_penalty:
            return disc_loss
        
        # Sample intermediate points between x and G_z where gradient norm penalty will be enforced 
        interpolated_shape = [len(x), 1, 1] if self.name == 'wavegan' else [len(x), 1, 1, 1]
        interpolates = x + tf.random.uniform(shape=interpolated_shape, minval=0., maxval=1.) * (G_z - x)
        
        with tf.GradientTape() as disc_tape:
            disc_tape.watch(interpolates)
            disc_interp = self.discriminator(interpolates)
        
        gradients = disc_tape.gradient(disc_interp, [interpolates])[0]
        g_norm = tf.math.sqrt(1e-9 + tf.math.reduce_sum(tf.math.square(gradients), axis=(1, 2) if self.name == 'wavegan' else (1, 2, 3)))
        
        # Add gradient penalty to the original loss
        disc_loss += 10 * tf.math.reduce_mean((g_norm - 1.) ** 2.)
        
        return disc_loss

    def generator_loss(self, D_G_z):
        """
        Method to compute the generator loss
        :param D_G_z:     Fake audio samples
        :return:          Generator loss
        """
        return -tf.math.reduce_mean(D_G_z)

#     @tf.function
    def train_step(self, x, gsteps=1, dsteps=1, gradient_penalty=True):
        """
        Method to perform one training step for this gan
        :param x:           Current batch data
        :return:            (generator loss, discriminator loss)
        """

        for ds, gs in product(range(dsteps), range(gsteps)):
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                z = tf.random.normal([len(x), self.latent_dim])

                G_z = self.generator(z, training=True)
                D_x = self.discriminator(x, training=True)
                D_G_z = self.discriminator(G_z, training=True)

                gen_loss = self.generator_loss(D_G_z)
                disc_loss = self.discriminator_loss(x, G_z, D_x, D_G_z, gradient_penalty)

            grad_gen = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
            grad_dis = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
            
#             nan_dis = [np.isnan(x).mean() for x in grad_dis]
#             nan_gen = [np.isnan(x).mean() for x in grad_gen]
#             stop = False
            
#             if any(x > 0 for x in nan_dis):
#                 stop = True
#                 print(f'{ds}{gs}:NaNs in dis grads: ', {var.name: nd for nd, var in zip(nan_dis, self.discriminator.trainable_variables)})
            
#             if any(x > 0 for x in nan_gen):
#                 stop = True
#                 print(f'{ds}{gs}:NaNs in gen grads: ', {var.name: nd for nd, var in zip(nan_gen, self.generator.trainable_variables)})
                
#             if stop:
#                 raise ValueError('Stopping: nan grads')
                
            if gs == 0:
#                 print('d')
                self.discriminator_optimizer.apply_gradients(zip(grad_dis, self.discriminator.trainable_variables))
            if ds == 0:
#                 print('g')
                self.generator_optimizer.apply_gradients(zip(grad_gen, self.generator.trainable_variables))

        return gen_loss, disc_loss

    def train(self, train_data, epochs, batch, gsteps=1, dsteps=1, gradient_penalty=True, preview_interval=1):
        """
        Method to train a gan
        :param train_data:          Training data pipeline
        :param epochs:              Number of training epochs
        :param batch:               Size of a training batch
        """        
        print('', end='', flush=True)

        with tqdm(total=epochs, desc=f'{type(self).__name__}: gs={gsteps} ds={dsteps} gp={gradient_penalty}', ncols=140) as pbar:
            
            for epoch in range(epochs):
                # tf.keras.backend.set_learning_phase(1)

                gen_losses = []
                disc_losses = []
                d_real = []
                d_fake = []
                accuracy = []
                
                for step, batch_data in enumerate(train_data):
                    # Training step
                    gen_loss, disc_loss = self.train_step(batch_data, gsteps, dsteps, gradient_penalty)
                    
                    # Log current losses
                    gen_losses.append(gen_loss.numpy())
                    disc_losses.append(disc_loss.numpy())

                    # Log discriminator output for real and fake samples
                    d_r = self.discriminator(batch_data, training=False)
                    d_f = self.discriminator(self.sample(len(batch_data)), training=False)
                    d_real.append(np.mean(d_r))
                    d_fake.append(np.mean(d_f))
                    acc = 0.5 * np.mean(d_r >= 0.5) + 0.5 * np.mean(d_f < 0.5)
                    accuracy.append(acc)
                                        
                    # Update progress bar
                    pbar.set_postfix(epoch=epoch+1, gen=np.mean(gen_losses).round(5), disc=np.mean(disc_losses).round(5), acc=np.mean(accuracy).round(2))
                    
                pbar.update(1)
                    
                self.performance['gen'].append(np.mean(gen_losses))
                self.performance['disc'].append(np.mean(disc_losses))
                self.performance['accuracy'].append(np.mean(accuracy))
                self.prediction_history['real'].append(np.mean(d_real))
                self.prediction_history['fake'].append(np.mean(d_fake))
                
                if epoch % preview_interval == 0:
                    self.preview(save=True, epoch=epoch)
                    self.show_progress(True)
                    
                self.save()
    
    def sample(self, n=1):
        z = tf.random.normal([n, self.latent_dim])
        return self.generator(z) 
                
    def preview(self, n=9, save=False, epoch=0):
        # tf.keras.backend.set_learning_phase(0)
        predictions = self.generator(tf.random.normal([n, self.latent_dim]), training=False).numpy()

        if predictions.ndim == 4:
            fig = plotting.imsc(predictions)
        else:
            fig = plotting.waveforms(predictions, spectrums=True)
            
        if save:            
            if not os.path.exists(os.path.join(self.dir, self.version)):
                os.makedirs(os.path.join(self.dir, self.version))
        
            filename = os.path.join(self.dir, self.version, f'preview_{epoch:04d}.jpg')
            plt.savefig(filename, bbox_inches='tight', quality=80)
            plt.close()
            return filename
        
        else:
            return fig

    def show_progress(self, save=False):
        fig, axes = plt.subplots(1, 3, figsize=(20, 3))
        axes[0].plot(self.performance['gen'])
        axes[0].plot(self.performance['disc'])
        axes[0].legend(['g', 'd'])
        axes[0].set_title('losses')

        axes[1].plot(self.performance['accuracy'])
        axes[1].set_title('accuracy')

        axes[2].plot(self.prediction_history['real'])
        axes[2].plot(self.prediction_history['fake'])
        axes[2].legend(['real', 'fake'])
        axes[2].set_title('discriminator on')
        
        if save:
            filename = os.path.join(self.dir, self.version, f'progress.png')
            fig.savefig(filename, bbox_inches='tight')
            plt.close()
            return filename
        else:
            return fig
