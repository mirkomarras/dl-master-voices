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
            self.input_s = tf.identity(get_tf_spectrum(self.input_a, self.sample_rate, self.frame_size, self.frame_stride, self.num_fft), name='input_s')

            with tf.variable_scope('conv1'):
                conv1_1 = tf.keras.layers.Conv2D(filters=96, kernel_size=[7, 7], strides=[2, 2], padding='SAME', reuse=self.reuse, name='cc1')(self.input_s)
                conv1_1 = tf.keras.layers.BatchNormalization(training=self.phase, name='bbn1', reuse=self.reuse)(conv1_1)
                conv1_1 = tf.nn.relu(conv1_1)
                conv1_1 = tf.keras.layers.MaxPool2D(pool_size=[3, 3], strides=[2, 2], name='mpool1')(conv1_1)

            with tf.variable_scope('conv2'):
                conv2_1 = tf.keras.layers.Conv2D(filters=256, kernel_size=[5, 5], strides=[2, 2], padding='SAME', reuse=self.reuse, name='cc2')(conv1_1)
                conv2_1 = tf.keras.layers.BatchNormalization(conv2_1, training=self.phase, name='bbn2', reuse=self.reuse)
                conv2_1 = tf.nn.relu(conv2_1)
                conv2_1 = tf.keras.layers.MaxPool2D(pool_size=[3, 3], strides=[2, 2], name='mpool2')(conv2_1)

            with tf.variable_scope('conv3'):
                conv3_1 = tf.keras.layers.Conv2D(filters=384, kernel_size=[3, 3], strides=[1, 1], padding='SAME', reuse=self.reuse, name='cc3_1')(conv2_1)
                conv3_1 = tf.keras.layers.BatchNormalization(training=self.phase, name='bbn3', reuse=self.reuse)(conv3_1)
                conv3_1 = tf.nn.relu(conv3_1)

                conv3_2 = tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding='SAME', reuse=self.reuse, name='cc3_2')(conv3_1)
                conv3_2 = tf.keras.layers.BatchNormalization(training=self.phase, name='bbn4', reuse=self.reuse)(conv3_2)
                conv3_2 = tf.nn.relu(conv3_2)

                conv3_3 = tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding='SAME', reuse=self.reuse, name='cc3_3')(conv3_2)
                conv3_3 = tf.keras.layers.BatchNormalization(training=self.phase, name='bbn5', reuse=self.reuse)(conv3_3)
                conv3_3 = tf.nn.relu(conv3_3)
                conv3_3 = tf.keras.layers.MaxPool2D(pool_size=[5, 3], strides=[3, 2], name='mpool3')(conv3_3)
                self.conv3_3 = conv3_3

            with tf.variable_scope('conv4'):
                conv4_3 = tf.keras.layers.Conv2D(filters=4096, kernel_size=[9, 1], strides=[1, 1], padding='VALID', reuse=self.reuse, name='cc4_1')(conv3_3)
                conv4_3 = tf.keras.layers.BatchNormalization(training=self.phase, name='bbn6', reuse=self.reuse)(conv4_3)
                conv4_3 = tf.nn.relu(conv4_3)
                conv4_3 = tf.reduce_mean(conv4_3, axis=[1, 2], name='apool4')
                conv4_3 = tf.nn.dropout(conv4_3, 0.5)

            with tf.variable_scope('fc'):
                flattened = tf.contrib.layers.flatten(conv4_3)
                flattened = tf.nn.l2_normalize(flattened)
                w = tf.Variable(tf.truncated_normal([4096, 1024], stddev=0.1), name='w')
                b = tf.Variable(tf.constant(0.1, shape=[1024]), name='b')
                h = tf.nn.xw_plus_b(flattened, w, b, name='scores')
                h = tf.nn.relu(h, name='relu')
                with tf.name_scope('dropout'):
                    h = tf.nn.dropout(h, self.dropout_keep_prob)

            self.embedding = self.graph.get_tensor_by_name('fc/scores:0')[0]

            with tf.variable_scope('output'):
                w = tf.get_variable('w', shape=[1024, n_classes], initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.constant(0.1, shape=[n_classes]), name='b')
                scores = tf.nn.xw_plus_b(h, w, b, name='scores')
                predictions = tf.argmax(scores, 1, name='predictions', output_type=tf.int32)

            self.logits = scores
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses, name='loss')

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss, name='optimizer')

            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(predictions, self.input_y)
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')

        if not self.reuse:
            with tf.Session(graph=self.graph, config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
                print('>', 'initializing', self.name, 'graph')
                sess.run(tf.global_variables_initializer())
                self.save(sess)

        print('>', self.name, 'built finished')
