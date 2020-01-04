#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.metrics import roc_curve, auc
from scipy import spatial
import tensorflow as tf
import numpy as np
import random
import time
import os

from helpers.audio import decode_audio, play_n_rec

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Model(object):
    """
       Class to represent Speaker Verification (SV) models with model saving / loading and playback & recording capabilities
    """

    def __init__(self, name='', id=-1, noises=None, cache=None, n_seconds=3, sample_rate=16000):
        """
        Method to initialize a speaker verification model that will be saved in 'data/pt_models/{name}'
        :param name:        String id for this model
        :param id:          Version id for this model - default: auto-increment value along the folder 'data/pt_models/{name}'
        :param noises:      Dictionary of paths to noise audio samples, e.g., noises['room'] = ['xyz.wav', ...]
        :param cache:       Dictionary of noise audio samples, e.g., cache['xyz.wav'] = [0.1, .54, ...]
        :param n_seconds:   Maximum number of seconds of an audio sample to be processed
        :param sample_rate: Sample rate of an audio sample to be processed
        """
        self.noises = noises
        self.cache = cache

        self.sample_rate=sample_rate
        self.n_seconds = n_seconds

        self.name = name
        self.dir = os.path.join('.', 'data', 'pt_models', self.name)
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        self.id = str(len(os.listdir(self.dir))) if id < 0 else id
        if not os.path.exists(os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id)))):
            os.makedirs(os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id))))

    def get_model(self):
        return self.inference_model

    def build(self, classes=None):
        """
        Method to build a speaker verification model that takes audio samples of shape (None, 1) and impulse flags (None, 3)
        :param classes:     Number of classes that this model should manage during training
        :return:            None
        """
        self.model = None
        self.inference = None
        self.classes = classes

    def save(self):
        """
        Method to save the weights of this model in 'data/pt_models/{name}/v{id}/model_weights.tf'
        :return:            None
        """
        print('>', 'saving', self.name, 'model')
        if not os.path.exists(os.path.join(self.dir)):
            os.makedirs(os.path.join(self.dir))
        self.model.save_weights(os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id)), 'model_weights.tf'))
        print('>', 'saved', self.name, 'model in', os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id)), 'model_weights.tf'))

    def load(self):
        """
        Method to load weights for this model from 'data/pt_models/{name}/v{id}/model_weights.tf'
        :return:            None
        """
        print('>', 'loading', self.name, 'model')
        if os.path.exists(os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id)))):
            if os.path.exists(os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id)), 'model_weights.tf')):
                self.model.load_weights(os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id)), 'model_weights.tf'))
                print('>', 'loaded weights from', os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id)), 'model_weights.tf'))
            else:
                print('>', 'no pre-trained weights for', self.name, 'model from', os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id)), 'model_weights.tf'))
        else:
            print('>', 'no directory for', self.name, 'model at', os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id))))

    def embed(self, signal):
        """
        Method to compute the embedding vector extracted by this model from signal with no playback & recording
        :param signal:      The audio signal from which the embedding vector will be extracted - shape (None,1)
        :return:            None
        """
        return self.inference_model.predict([np.expand_dims(signal, axis=0), np.expand_dims(np.zeros(3), axis=0)])

    def train(self, train_data, test_data, steps_per_epoch=10, epochs=1, learning_rate=1e-1, patience=5):
        """
        Method to train and validate this model
        :param train_data:      Training data pipeline - shape ({'input_1': (batch, None, 1), 'input_2': (batch, 3)}), (batch, classes)
        :param test_data:       Pre-computed testing data pairs - shape ((pairs, None, 1), (pairs, None, 1)), (pairs, binary_label)
        :param steps_per_epoch: Number of steps per epoch
        :param epochs:          Number of training epochs
        :param learning_rate:   Learning rate
        :param patience:        Number of epochs with non-improving EER willing to wait
        :return:                None
        """
        print('>', 'training', self.name, 'model')
        self.model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
        num_nonimproving_steps, last_eer = 0, 1.0
        for epoch in range(epochs):
            tf.keras.backend.set_learning_phase(1)
            self.model.fit(train_data, steps_per_epoch=steps_per_epoch, initial_epoch=epoch, epochs=epoch+1)
            eer, _, _ = self.test(test_data)
            if eer < last_eer:
                print('>', 'eer improved from', round(last_eer, 2), 'to', round(eer, 2))
                num_nonimproving_steps = 0
                last_eer = eer
                self.save()
            else:
                print('>', 'eer NOT improved from', round(last_eer, 2))
                num_nonimproving_steps += 1
            if num_nonimproving_steps == patience:
                print('>', 'early stopping training after', num_nonimproving_steps, 'non-improving steps')
                break
        print('>', 'trained', self.name, 'model')

    def test(self, test_data):
        """
        Method to test this model against verification attempts
        :param test_data:       Pre-computed testing data pairs - shape ((pairs, None, 1), (pairs, None, 1)), (pairs, binary_label)
        :return:                (Model EER, EER threshold, FAR1% threshold)
        """
        print('>', 'testing', self.name, 'model')
        (x1, x2), y = test_data
        eer, thr_eer, thr_far1 = 0, 0, 0
        similarity_scores = np.zeros(len(x1))
        tf.keras.backend.set_learning_phase(0)
        for pair_id, (f1, f2) in enumerate(zip(x1, x2)):
            similarity_scores[pair_id] = 1 - spatial.distance.cosine(self.embed(f1), self.embed(f2))
            if pair_id > 2:
                far, tpr, thresholds = roc_curve(y[:pair_id+1], similarity_scores[:pair_id+1])
                frr = 1 - tpr
                id_eer = np.argmin(np.abs(far - frr))
                id_far1 = np.argmin(np.abs(far - 0.01))
                eer = float(np.mean([far[id_eer], frr[id_eer]]))
                thr_eer = thresholds[id_eer]
                thr_far1 = thresholds[id_far1]
                print('\r> pair %5.0f / %5.0f - eer: %3.5f - thr@eer: %3.5f - thr@far1: %3.1f' % (pair_id+1, len(x1), eer, thr_eer, thr_far1), end='')
        print()
        print('>', 'tested', self.name, 'model')
        return eer, thr_eer, thr_far1

    def impersonate(self, impostor_signal, threshold, policy, x_mv_test, y_mv_test, male_x_mv_test, female_x_mv_test, n_templates=10):
        """
        Method to test this model under impersonation attempts
        :param impostor_signal:     Audio signal against which this model is tested - shape (None, 1)
        :param threshold:           Verification threshold
        :param policy:              Verification policy - choices ['avg', 'any']
        :param x_mv_test:           Testing users' audio samples - shape (users, n_templates, None, 1)
        :param y_mv_test:           Testing users' labels - shape (users, n_templates)
        :param male_x_mv_test:      Male users' ids
        :param female_x_mv_test:    Female users' ids
        :param n_templates:         Number of audio samples to create a user template
        :return:                    {'m': impersonation rate against male users, 'f': impersonation rate against female users}
        """
        print('>', 'impersonating', self.name, 'model')
        mv_emb = self.embed(impostor_signal)
        mv_fac = np.zeros(len(np.unique(y_mv_test)))
        for class_index, class_label in enumerate(np.unique(y_mv_test)):
            template = [self.embed(signal) for signal in x_mv_test[class_index*n_templates:(class_index+1)*n_templates]]
            if policy == 'any':
                mv_fac[class_index] = len([1 for template_emb in np.array(template) if 1 - spatial.distance.cosine(template_emb, mv_emb) > threshold])
            elif policy == 'avg':
                mv_fac[class_index] = 1 if 1 - spatial.distance.cosine(mv_emb, np.mean(np.array(template), axis=0)) > threshold else 0
        print('>', 'impersonated', self.name, 'model')
        return {'m': len([index for index, fac in enumerate(mv_fac) if fac > 0 and index in male_x_mv_test]) / len(male_x_mv_test), 'f': len([index for index, fac in enumerate(mv_fac) if fac > 0 and index in female_x_mv_test]) / len(female_x_mv_test)}