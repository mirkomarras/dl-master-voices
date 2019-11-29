#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import queue
import time
import os
import random

from src.models.verifier.tf_keras.model import Model
from src.helpers.audio import get_tf_spectrum

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class VggVox(Model):

    def __init__(self, graph=tf.Graph(), var2std_epsilon=0.00001, reuse=False, id=''):
        super().__init__(graph, var2std_epsilon, reuse, id)
        self.name = 'vggvox'
        self.id = self.get_version_id()

    def build(self, input_x, input_y, n_classes=0, n_filters=24, noises=None, cache=None, augment=0, n_seconds=3, sample_rate=16000):

        with self.graph.as_default():
            print('>', 'building', self.name, 'model')

            super().build(input_x, input_y, n_classes, n_filters, noises, cache, augment, n_seconds, sample_rate)
            self.input_s = tf.identity(get_tf_spectrum(self.input_a, self.sample_rate, self.frame_size, self.frame_stride, self.num_fft))

            with tf.variable_scope('conv1'):
                conv1_1 = tf.keras.layers.Conv2D(filters=96, kernel_size=[7, 7], strides=[2, 2], padding='SAME', name='cc1')(self.input_s)
                conv1_1 = tf.keras.layers.BatchNormalization(name='bbn1')(conv1_1)
                conv1_1 = tf.keras.layers.ReLU()(conv1_1)
                conv1_1 = tf.keras.layers.MaxPool2D(pool_size=[3, 3], strides=[2, 2], name='mpool1')(conv1_1)

            with tf.variable_scope('conv2'):
                conv2_1 = tf.keras.layers.Conv2D(filters=256, kernel_size=[5, 5], strides=[2, 2], padding='SAME', name='cc2')(conv1_1)
                conv2_1 = tf.keras.layers.BatchNormalization(conv2_1, name='bbn2')
                conv2_1 = tf.keras.layers.ReLU()(conv2_1)
                conv2_1 = tf.keras.layers.MaxPool2D(pool_size=[3, 3], strides=[2, 2], name='mpool2')(conv2_1)

            with tf.variable_scope('conv3'):
                conv3_1 = tf.keras.layers.Conv2D(filters=384, kernel_size=[3, 3], strides=[1, 1], padding='SAME', name='cc3_1')(conv2_1)
                conv3_1 = tf.keras.layers.BatchNormalization(name='bbn3')(conv3_1)
                conv3_1 = tf.keras.layers.ReLU()(conv3_1)

                conv3_2 = tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding='SAME', name='cc3_2')(conv3_1)
                conv3_2 = tf.keras.layers.BatchNormalization(name='bbn4')(conv3_2)
                conv3_2 = tf.keras.layers.ReLU()(conv3_2)

                conv3_3 = tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding='SAME', name='cc3_3')(conv3_2)
                conv3_3 = tf.keras.layers.BatchNormalization(name='bbn5')(conv3_3)
                conv3_3 = tf.keras.layers.ReLU()(conv3_3)
                conv3_3 = tf.keras.layers.MaxPool2D(pool_size=[5, 3], strides=[3, 2], name='mpool3')(conv3_3)
                self.conv3_3 = conv3_3

            with tf.variable_scope('conv4'):
                conv4_3 = tf.keras.layers.Conv2D(filters=4096, kernel_size=[9, 1], strides=[1, 1], padding='VALID', name='cc4_1')(conv3_3)
                conv4_3 = tf.keras.layers.BatchNormalization(name='bbn6')(conv4_3)
                conv4_3 = tf.keras.layers.ReLU()(conv4_3)
                conv4_3 = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=[1, 2], name='apool4'))(conv4_3)
                conv4_3 = tf.keras.layers.Dropout(0.5)(conv4_3)

            with tf.variable_scope('fc'):
                flattened = tf.keras.layers.Flatten()(conv4_3)
                flattened = tf.keras.backend.l2_normalize()(flattened)
                h = tf.keras.layers.Dense(1024)(flattened)
                h = tf.keras.layers.ReLU(name='relu')(h)
                with tf.name_scope('dropout'):
                    h = tf.keras.layers.Dropout(self.dropout_keep_prob)(h)

            self.embedding = self.graph.get_tensor_by_name('fc/scores:0')[0]

            with tf.variable_scope('output'):
                output = tf.keras.layers.Dense(n_classes, activation='softmax', name='output')(h)

            self.model = tf.keras.Model(inputs=[self.inpux_x, self.input_y, self.speaker, self.room, self.microphone], outputs=[output])
