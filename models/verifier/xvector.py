#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import os
import random

from models.verifier.model import Model
from helpers.audio import get_tf_filterbanks

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def normalize_with_moments(x):
    tf_mean, tf_var = tf.nn.moments(x, 1)
    x = tf.concat([tf_mean, tf.sqrt(tf_var + 0.00001)], 1)
    return x

class XVector(Model):

    """
       Class to represent Speaker Verification (SV) model based on the XVector architecture - Embedding vectors of size 512 are returned
       Snyder, D., Garcia-Romero, D., Sell, G., Povey, D., & Khudanpur, S. (2018, April).
       X-vectors: Robust dnn embeddings for speaker recognition.
       In: 2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 5329-5333. IEEE.
    """

    def __init__(self, name='xvector', id=''):
        super().__init__(name, id)
        self._uses_spectrum = False


    def compute_acoustic_representation(self, e):
        return get_tf_filterbanks(e, n_filters=24)


    def build(self, classes=0, embs_name='embs', embs_size=512, loss='softmax', aggregation='gvlad', vlad_clusters=12, ghost_clusters=2, weight_decay=1e-3, mode='train'):
        super().build(classes, embs_name, embs_size, loss, aggregation, vlad_clusters, ghost_clusters, weight_decay, mode)
        print('>', 'building', self.name, 'model on', classes, 'classes')

        input_layer = tf.keras.Input(shape=(None, 24,), name='Input_1')

        # Layer parameters
        layer_sizes = [512, 512, 512, 512, 3 * 512]
        kernel_sizes = [5, 5, 7, 1, 1]

        # Frame information layer
        x = input_layer
        for i, (kernel_size, layer_size) in enumerate(zip(kernel_sizes, layer_sizes)):
            x = tf.keras.layers.Conv1D(kernel_size=kernel_size, filters=layer_size, padding='SAME')(x)
            x = tf.keras.layers.ReLU()(x)
            x = tf.keras.layers.BatchNormalization(epsilon=1e-3, gamma_initializer=tf.constant_initializer(1.0), beta_initializer=tf.constant_initializer(0.0))(x)
            if i != len(kernel_sizes) - 1:
                x = tf.keras.layers.Dropout(0.1)(x)

        # Statistic pooling
        x = tf.keras.layers.Lambda(lambda x: normalize_with_moments(x))(x)

        # Embedding layers
        x = tf.keras.layers.Dense(embs_size, name='fc1')(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.BatchNormalization(epsilon=1e-3, gamma_initializer=tf.constant_initializer(1.0), beta_initializer=tf.constant_initializer(0.0))(x)
        x = tf.keras.layers.Dropout(0.1)(x)

        x = tf.keras.layers.Dense(embs_size, name='embs')(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.BatchNormalization(epsilon=1e-3, gamma_initializer=tf.constant_initializer(1.0), beta_initializer=tf.constant_initializer(0.0))(x)

        output_layer = tf.keras.layers.Dense(classes, activation='softmax', kernel_initializer='orthogonal', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(weight_decay), bias_regularizer=tf.keras.regularizers.l2(weight_decay))(x)

        self.embs_name = embs_name
        self.model = tf.keras.Model(inputs=input_layer, outputs=output_layer, name='xvector_{}_{}'.format(loss, aggregation))

        print('>', 'built', self.name, 'model on', classes, 'classes')