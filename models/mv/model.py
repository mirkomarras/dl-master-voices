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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class MasterVocoder(object):
    """
       Class to represent Master Voice (MV) models with master voice training and testing functionalities
    """

    def __init__(self, sample_rate, dir_mv, dir_sv):
        """
        Method to initialize a master voice model that will save audio samples in 'data/vs_mv_data/{net}-{netv}_{gan}-{ganv}_{f|m}-{f|m}_{mv|sv}'
        :param sample_rate:     Sample rate of an audio sample to be processed
        :param dir_mv:          Path to the folder where master voice audio samples will be saved
        :param dir_sv:          Path to the folder where original gan audio samples will be saved
        """
        self.sample_rate = sample_rate
        self.dir_mv = dir_mv
        self.dir_sv = dir_sv
        if not os.path.exists(self.dir_mv) or not os.path.exists(self.dir_sv):
            os.makedirs(self.dir_mv)
            os.makedirs(self.dir_sv)
        self.id_mv = str(len(os.listdir(self.dir_mv)))
        self.id_sv = str(len(os.listdir(self.dir_sv)))
        assert self.id_mv == self.id_sv

    def set_generator(self, gan):
        """
        Method to set the generator of fake audio samples
        :param gan:     Gan model from which the generator is taken
        """
        gan.build()
        self.gan = gan

    def set_verifier(self, verifier):
        """
        Method to set the verifier against which master voices are optimized
        :param verifier:    Verifier model
        """
        verifier.build(classes=2)
        self.verifier = verifier

    def build(self):
        """
        Method to create a vocoder: one branch generates fake gan samples, the other branch received real audio samples
        """
        embedding_1 = self.verifier.get_model()(self.verifier.get_model().input)
        embedding_2 = self.verifier.get_model()([self.gan.get_generator().output, self.verifier.get_model().input[1]])
        similarity = tf.keras.layers.Dot(axes=1, normalize=True)([embedding_1, embedding_2])
        self.vocoder = tf.keras.Model([self.gan.get_generator().input, self.verifier.get_model().input[0], self.verifier.get_model().input[1]], similarity)

    def get_vocoder(self):
        """
        Method to get the vocoder built by this class
        :return:    The vocoder built by this class
        """
        return self.vocoder

    def train(self, train_data, n_iterations, n_epochs, n_steps_per_epoch, min_val=1e-5, min_sim=0.25, max_sim=1.00, learning_rate=1e-1, mv_test_thrs=None, mv_test_data=None):
        """
        Method to train master voice samples
        :param train_data:          Real audio data against which master voices are optimized - shape (None,1)
        :param n_iterations:        Number of master voice samples to be created
        :param n_epochs:            Number of epochs for each master voice sample
        :param n_steps_per_epoch:   Steps per epoch for each master voice samples
        :param min_val:             Minimum value for a pertubation
        :param min_sim:             Minimum value for considering a gradient
        :param max_sim:             Maximum value for considering a gradient
        :param learning_rate:       Learning rate
        :param mv_test_thrs:        Thresholds for eer and far1 for the embedded verifier
        :param mv_test_data:        Real user against which the current master voice are validated
        """
        filter_gradients = lambda c, g, t1, t2: [g[i] for i in range(len(c)) if c[i] >= t1 and c[i] <= t2]

        impulse = np.zeros(3).astype(np.float32)
        for iter in range(n_iterations):
            latent_mv = np.random.normal(size=(1, 100)).astype(np.float32)
            latent_sv = np.copy(latent_mv)
            for epoch in range(n_epochs):
                for step, batch_data in enumerate(train_data):

                    input_1 = tf.Variable(np.tile(latent_mv, (len(batch_data), 1)), dtype=tf.float32)
                    input_2 = tf.Variable(batch_data, dtype=tf.float32)
                    input_3 = tf.Variable(np.tile(impulse, (len(batch_data), 1)), dtype=tf.float32)
                    with tf.GradientTape() as tape:
                        loss = self.vocoder([input_1, input_2, input_3])
                    grads = tape.gradient(loss, input_1)

                    filtered_grads = filter_gradients(loss, grads, min_sim, max_sim)

                    if len(filtered_grads) > 0:
                        perturbation = np.mean(filtered_grads, axis=0) * learning_rate
                        perturbation = np.clip(perturbation, min_val, None)
                        latent_mv += perturbation

                    print('\rIter ', iter+1, 'of', n_iterations, 'Epoch', epoch+1, 'of', n_epochs, 'Step', step+1, 'of', n_steps_per_epoch, 'loss', round(np.mean(loss), 5), end='')
                    if mv_test_thrs is not None and mv_test_data is not None:
                        eer_results, far1_results = self.test(latent_mv, mv_test_thrs, mv_test_data)
                        print('eer_imp', (eer_results['m'], eer_results['f']), 'far1_imp', (far1_results['m'], far1_results['f']), end='')

                self.save(iter, latent_sv, latent_mv)

    def save(self, iter, latent_sv, latent_mv):
        """
        Method to save original and optimized master voices
        :param iter:        Number of the current iteration
        :param latent_sv:   Original latent vector
        :param latent_mv:   Optimized latent vector
        """
        if not os.path.exists(os.path.join(self.dir_mv, 'v' + str('{:03d}'.format(self.id_mv)))) or not os.path.exists(os.path.join(self.dir_sv, 'v' + str('{:03d}'.format(self.id_sv)))):
            os.makedirs(os.path.join(self.dir_mv, 'v' + str('{:03d}'.format(self.id_mv))))
            os.makedirs(os.path.join(self.dir_sv, 'v' + str('{:03d}'.format(self.id_sv))))
        np.save(os.path.join(self.dir_mv, 'v' + str('{:03d}'.format(self.id_mv)), 'sample_' + str(iter) + '.npz'), latent_mv)
        print('>', 'saved mv latent in', os.path.join(self.dir_mv, 'v' + str('{:03d}'.format(self.id_mv)), 'sample_' + str(iter) + '.npz'))
        sf.write(os.path.join(self.dir_mv, 'v' + str('{:03d}'.format(self.id_mv)), 'sample_' + str(iter) + '.wav'), self.gan.get_generator()(latent_mv).numpy(), self.sample_rate)
        print('>', 'saved mv wav in', os.path.join(self.dir_mv, 'v' + str('{:03d}'.format(self.id_mv)), 'sample_' + str(iter) + '.wav'))
        np.save(os.path.join(self.dir_sv, 'v' + str('{:03d}'.format(self.id_sv)), 'sample_' + str(iter) + '.npz'), latent_sv)
        print('>', 'saved sv latent in', os.path.join(self.dir_sv, 'v' + str('{:03d}'.format(self.id_sv)), 'sample_' + str(iter) + '.npz'))
        sf.write(os.path.join(self.dir_sv, 'v' + str('{:03d}'.format(self.id_sv)), 'sample_' + str(iter) + '.wav'), self.gan.get_generator()(latent_sv).numpy(), self.sample_rate)
        print('>', 'saved sv wav in', os.path.join(self.dir_sv, 'v' + str('{:03d}'.format(self.id_sv)), 'sample_' + str(iter) + '.wav'))

    def test(self, latent, mv_test_thrs, mv_test_data):
        """
        Method to test the current master voice
        :param latent:          Latent vector to be tested
        :param mv_test_thrs:    Thresholds for eer and far1 for the embedded verifier
        :param mv_test_data:    Real user against which the current master voice are validated
        :return:
        """
        _, thr_eer, thr_far1 = mv_test_thrs
        x_mv_test, y_mv_test, male_x_mv_test, female_x_mv_test = mv_test_data
        eer_results = self.verifier.impersonate(self.gan.get_generator()(latent).numpy(), thr_eer, 'any', x_mv_test, y_mv_test, male_x_mv_test, female_x_mv_test, args.n_templates)
        far1_results = self.verifier.impersonate(self.gan.get_generator()(latent).numpy(), thr_far1, 'any', x_mv_test, y_mv_test, male_x_mv_test, female_x_mv_test, args.n_templates)
        return eer_results, far1_results
