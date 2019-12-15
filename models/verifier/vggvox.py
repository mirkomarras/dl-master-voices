#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import queue
import time
import os
import random

from models.verifier.model import Model
from helpers.audio import play_n_rec, get_tf_spectrum

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class VggVox(Model):

    """
       Class to represent Speaker Verification (SV) model based on the VGG16 architecture - Embedding vectors of size 1024 are returned
       Nagrani, A., Chung, J. S., & Zisserman, A. (2017).
       VoxCeleb: A Large-Scale Speaker Identification Dataset.
       Proc. Interspeech 2017, 2616-2620.
    """

    def __init__(self, name='vggvox', id=-1, noises=None, cache=None, n_seconds=3, sample_rate=16000):
        super().__init__(name, id, noises, cache, n_seconds, sample_rate)

    def build(self, classes=None):
        super().build(classes)
        print('>', 'building', self.name, 'model on', classes, 'classes')

        signal_input = tf.keras.Input(shape=(None,1,))
        impulse_input = tf.keras.Input(shape=(3,))

        x = tf.keras.layers.Lambda(lambda x: play_n_rec(x, self.noises, self.cache), name='play_n_rec')([signal_input, impulse_input])
        x = tf.keras.layers.Lambda(lambda x: get_tf_spectrum(x, self.sample_rate), name='acoustic_layer')(x)

        x = tf.keras.layers.Conv2D(filters=96, kernel_size=[7, 7], strides=[2, 2], padding='SAME', name='cc1')(x)
        x = tf.keras.layers.BatchNormalization(name='bbn1')(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPool2D(pool_size=[3, 3], strides=[2, 2], name='mpool1')(x)

        x = tf.keras.layers.Conv2D(filters=256, kernel_size=[5, 5], strides=[2, 2], padding='SAME', name='cc2')(x)
        x = tf.keras.layers.BatchNormalization(name='bbn2')(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPool2D(pool_size=[3, 3], strides=[2, 2], name='mpool2')(x)

        x = tf.keras.layers.Conv2D(filters=384, kernel_size=[3, 3], strides=[1, 1], padding='SAME', name='cc3_1')(x)
        x = tf.keras.layers.BatchNormalization(name='bbn3')(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding='SAME', name='cc3_2')(x)
        x = tf.keras.layers.BatchNormalization(name='bbn4')(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding='SAME', name='cc3_3')(x)
        x = tf.keras.layers.BatchNormalization(name='bbn5')(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPool2D(pool_size=[5, 3], strides=[3, 2], name='mpool3')(x)

        x = tf.keras.layers.Conv2D(filters=4096, kernel_size=[9, 1], strides=[1, 1], padding='VALID', name='cc4_1')(x)
        x = tf.keras.layers.BatchNormalization(name='bbn6')(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=[1, 2], name='apool4'))(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Flatten()(x)

        embedding_layer = tf.keras.layers.Dense(1024, name='embedding_layer')(x)
        output = tf.keras.layers.Dense(classes, activation='softmax')(embedding_layer)

        self.model = tf.keras.Model(inputs=[signal_input, impulse_input], outputs=[output])
        print('>', 'built', self.name, 'model on', classes, 'classes')

        super().load()

        self.inference_model = tf.keras.Model(inputs=[signal_input, impulse_input], outputs=[embedding_layer])
        print('>', 'built', self.name, 'inference model')

        self.model.summary()