#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scipy.spatial.distance import euclidean, cosine
from keras.callbacks import LearningRateScheduler
from sklearn.metrics import roc_curve, auc
from scipy import spatial
import tensorflow as tf
import numpy as np
import random
import time
import os

from helpers.audio import decode_audio, play_n_rec

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class StepDecay():
    def __init__(self, init_alpha=0.01, decay_factor=0.25, decay_step=10):
        self.init_alpha = init_alpha
        self.decay_factor = decay_factor
        self.decay_step = decay_step

    def __call__(self, epoch):
        exp = np.floor((1 + epoch) / self.decay_step)
        alpha = self.init_alpha * (self.decay_factor ** exp)
        print('Learning rate for next epoch', float(alpha))
        return float(alpha)

class VladPooling(tf.keras.layers.Layer):
    def __init__(self, mode, k_centers, g_centers=0, **kwargs):
        self.k_centers = k_centers
        self.g_centers = g_centers
        self.mode = mode
        super(VladPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        self.cluster = self.add_weight(shape=[self.k_centers+self.g_centers, input_shape[0][-1]], name='centers', initializer='orthogonal')
        self.built = True

    def compute_output_shape(self, input_shape):
        assert input_shape
        return (input_shape[0][0], self.k_centers*input_shape[0][-1])

    def call(self, x):
        feat, cluster_score = x
        num_features = feat.shape[-1]

        max_cluster_score = tf.keras.backend.max(cluster_score, -1, keepdims=True)
        exp_cluster_score = tf.keras.backend.exp(cluster_score - max_cluster_score)
        A = exp_cluster_score / tf.keras.backend.sum(exp_cluster_score, axis=-1, keepdims = True)

        A = tf.keras.backend.expand_dims(A, -1)
        feat_broadcast = tf.keras.backend.expand_dims(feat, -2)
        feat_res = feat_broadcast - self.cluster
        weighted_res = tf.math.multiply(A, feat_res)
        cluster_res = tf.keras.backend.sum(weighted_res, [1, 2])

        if self.mode == 'gvlad':
            cluster_res = cluster_res[:, :self.k_centers, :]

        cluster_l2 = tf.keras.backend.l2_normalize(cluster_res, -1)
        outputs = tf.keras.backend.reshape(cluster_l2, [-1, int(self.k_centers) * int(num_features)])
        return outputs

class Model(object):
    """
       Class to represent Speaker Verification (SV) models with model saving / loading and playback & recording capabilities
    """

    def __init__(self, name='', id=-1, noises=None, cache=None, n_seconds=3, sample_rate=16000, emb_size=1024):
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
        self.emb_size = emb_size

        self.name = name
        self.dir = os.path.join('.', 'data', 'pt_models', self.name)
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        self.id = len(os.listdir(self.dir)) if id < 0 else id
        if not os.path.exists(os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id)))):
            os.makedirs(os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id))))

    def get_model(self):
        return self.inference_model

    def build(self, classes=None, loss='softmax', aggregation='avg', vlad_clusters=12, ghost_clusters=2, weight_decay=1e-4, augment=0):
        """
        Method to build a speaker verification model that takes audio samples of shape (None, 1) and impulse flags (None, 3)
        :param classes:         Number of classes that this model should manage during training
        :param loss:            Type of loss
        :param aggregation:     Type of aggregation function
        :param vlad_clusters:   Number of vlad clusters in vlad and gvlad
        :param ghost_clusters:  Number of ghost clusters in vlad and gvlad
        :param weight_decay:    Decay of weights in convolutional layers
        :param augment:         Augmentation flag
        :return:                None
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

    def train(self, train_data, test_data, steps_per_epoch=10, epochs=1, learning_rate=1e-1, patience=20, decay_factor=0.1, decay_step=10, optimizer='adam'):
        """
        Method to train and validate this model
        :param train_data:      Training data pipeline - shape ({'input_1': (batch, None, 1), 'input_2': (batch, 3)}), (batch, classes)
        :param test_data:       Pre-computed testing data pairs - shape ((pairs, None, 1), (pairs, None, 1)), (pairs, binary_label)
        :param steps_per_epoch: Number of steps per epoch
        :param epochs:          Number of training epochs
        :param learning_rate:   Learning rate
        :param patience:        Number of epochs with non-improving EER willing to wait
        :param decay_factor:    Decay in terms of learning rate
        :param decay_step:      Number of epoch for each decay in learning rate
        :param optimizer:       Type of training optimizer
        :return:                None
        """

        print('>', 'training', self.name, 'model')
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        schedule = StepDecay(init_alpha=learning_rate, decay_factor=decay_factor, decay_step=decay_step)
        saving = tf.keras.callbacks.ModelCheckpoint(os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id)), 'model_weights.tf'), monitor='loss', mode='min', save_best_only=True)
        earlystopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience)
        learning = LearningRateScheduler(schedule)
        callbacks = [learning, saving, earlystopping]
        self.model.fit(train_data, steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=callbacks)
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
            similarity_scores[pair_id] = (1 - cosine(self.embed(f1), self.embed(f2)) + 1) / 2
            if pair_id > 2:
                far, tpr, thresholds = roc_curve(y[:pair_id+1], similarity_scores[:pair_id+1], pos_label=1)
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