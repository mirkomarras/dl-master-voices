#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import queue
import time
import os
import random

from models.verifier.model import Model
from helpers.audio import play_n_rec, get_tf_filterbanks

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class XVector(Model):

    """
       Class to represent Speaker Verification (SV) model based on the XVector architecture - Embedding vectors of size 512 are returned
       Snyder, D., Garcia-Romero, D., Sell, G., Povey, D., & Khudanpur, S. (2018, April).
       X-vectors: Robust dnn embeddings for speaker recognition.
       In: 2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 5329-5333. IEEE.
    """

    def __init__(self, name='xvector', id='', noises=None, cache=None, n_seconds=3, sample_rate=16000):
        super().__init__(name, id, noises, cache, n_seconds, sample_rate)
        self.n_filters = 24

    def __normalize_with_moments(self, x):
        tf_mean, tf_var = tf.nn.moments(x, 1)
        x = tf.concat([tf_mean, tf.sqrt(tf_var + 0.00001)], 1)
        return x

    def build(self, classes=None):
        super().build(classes)
        print('>', 'building', self.name, 'model on', classes, 'classes')

        signal_input = tf.keras.Input(shape=(None,1,))
        impulse_input = tf.keras.Input(shape=(3,))

        x = tf.keras.layers.Lambda(lambda x: play_n_rec(x, self.noises, self.cache), name='play_n_rec')([signal_input, impulse_input])
        x = tf.keras.layers.Lambda(lambda x: get_tf_filterbanks(x, self.sample_rate), name='acoustic_layer')(x)

        # Layer parameters
        layer_sizes = [512, 512, 512, 512, 3 * 512]
        kernel_sizes = [5, 5, 7, 1, 1]
        embedding_sizes = [512, 512]

        # Frame information layer
        prev_dim = self.n_filters
        for i, (kernel_size, layer_size) in enumerate(zip(kernel_sizes, layer_sizes)):
            x = tf.nn.conv1d(x, tf.random.truncated_normal([kernel_size, prev_dim, layer_size], stddev=0.1), stride=1, padding='SAME')
            x = tf.nn.bias_add(x, tf.constant(0.1, shape=[layer_size]))
            x = tf.keras.layers.ReLU()(x)
            x = tf.keras.layers.BatchNormalization(epsilon=1e-3, gamma_initializer=tf.constant_initializer(1.0), beta_initializer=tf.constant_initializer(0.0))(x)
            prev_dim = layer_size
            if i != len(kernel_sizes) - 1:
                x = tf.keras.layers.Dropout(0.1)(x)

        # Statistic pooling
        x = tf.keras.layers.Lambda(lambda x: self.__normalize_with_moments(x))(x)

        # Embedding layers
        embedding_layers = []
        for i, out_dim in enumerate(embedding_sizes):
            x = tf.keras.layers.Dense(out_dim)(x)
            embedding_layers.append(x)
            x = tf.keras.layers.ReLU()(x)
            x = tf.keras.layers.BatchNormalization(epsilon=1e-3, gamma_initializer=tf.constant_initializer(1.0), beta_initializer=tf.constant_initializer(0.0))(x)
            if i != len(embedding_sizes) - 1:
                x = tf.keras.layers.Dropout(0.1)(x)

        output = tf.keras.layers.Dense(classes, activation='softmax')(x)

        self.model = tf.keras.Model(inputs=[signal_input, impulse_input], outputs=[output])
        print('>', 'built', self.name, 'model on', classes, 'classes')

        super().load()

        self.inference_model = tf.keras.Model(inputs=[signal_input, impulse_input], outputs=[embedding_layers[0]])
        print('>', 'built', self.name, 'inference model')

        self.model.summary()