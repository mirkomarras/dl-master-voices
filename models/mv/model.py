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

class MasterVocoder(object):

    def __init__(self, sample_rate, dir_mv, dir_sv):
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
        gan.build()
        self.gan = gan

    def set_verifier(self, verifier):
        verifier.build(classes=2)
        self.verifier = verifier

    def build(self):
        embedding_1 = self.verifier.get_model()(self.verifier.get_model().input)
        embedding_2 = self.verifier.get_model()([self.gan.get_generator().output, self.verifier.get_model().input[1]])
        similarity = tf.keras.layers.Dot(axes=1, normalize=True)([embedding_1, embedding_2])
        self.vocoder = tf.keras.Model([self.gan.get_generator().input, self.verifier.get_model().input[0], self.verifier.get_model().input[1]], similarity)

    def get_vocoder(self):
        return self.vocoder

    def train(self, train_data, n_iterations, n_epochs, n_steps_per_epoch, min_val=1e-5, min_sim=0.25, max_sim=0.75, learning_rate=1e-1, mv_test_thrs=None, mv_test_data=None):
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
                        eer_results, far1_results = self.test(mv_test_thrs, mv_test_data)
                        print('eer_imp', (eer_results['m'], eer_results['f']), 'far1_imp', (far1_results['m'], far1_results['f']), end='')

                self.save(iter, latent_sv, latent_mv)

    def save(self, iter, latent_sv, latent_mv):
        if not os.path.exists(os.path.join(self.dir_mv, 'v' + str(self.id_mv))) or not os.path.exists(os.path.join(self.dir_sv, 'v' + str(self.id_sv))):
            os.makedirs(os.path.join(self.dir_mv, 'v' + str(self.id_mv)))
            os.makedirs(os.path.join(self.dir_sv, 'v' + str(self.id_sv)))
        np.save(os.path.join(self.dir_mv, 'v' + str(self.id_mv), 'sample_' + str(iter) + '.npz'), latent_mv)
        print('>', 'saved mv latent in', os.path.join(self.dir_mv, 'v' + str(self.id_mv), 'sample_' + str(iter) + '.npz'))
        sf.write(os.path.join(self.dir_mv, 'v' + str(self.id_mv), 'sample_' + str(iter) + '.wav'), self.gan.get_generator()(latent_mv).numpy(), self.sample_rate)
        print('>', 'saved mv wav in', os.path.join(self.dir_mv, 'v' + str(self.id_mv), 'sample_' + str(iter) + '.wav'))
        np.save(os.path.join(self.dir_sv, 'v' + str(self.id_sv), 'sample_' + str(iter) + '.npz'), latent_sv)
        print('>', 'saved sv latent in', os.path.join(self.dir_sv, 'v' + str(self.id_sv), 'sample_' + str(iter) + '.npz'))
        sf.write(os.path.join(self.dir_sv, 'v' + str(self.id_sv), 'sample_' + str(iter) + '.wav'), self.gan.get_generator()(latent_sv).numpy(), self.sample_rate)
        print('>', 'saved sv wav in', os.path.join(self.dir_sv, 'v' + str(self.id_sv), 'sample_' + str(iter) + '.wav'))

    def test(self, mv_test_thrs, mv_test_data):
        _, thr_eer, thr_far1 = mv_test_thrs
        x_mv_test, y_mv_test, male_x_mv_test, female_x_mv_test = mv_test_data
        eer_results = self.verifier.impersonate(self.gan.get_generator()(latent_mv).numpy(), thr_eer, 'any', x_mv_test, y_mv_test, male_x_mv_test, female_x_mv_test, args.n_templates)
        far1_results = self.verifier.impersonate(self.gan.get_generator()(latent_mv).numpy(), thr_far1, 'any', x_mv_test, y_mv_test, male_x_mv_test, female_x_mv_test, args.n_templates)
        return eer_results, far1_results
