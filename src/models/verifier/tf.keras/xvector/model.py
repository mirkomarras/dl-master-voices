#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import queue
import time
import os
import random

from src.models.verifier.tf.model import Model
from src.helpers.audio import get_tf_mfccs

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class XVector(Model):

    def __init__(self, graph=tf.Graph(), var2std_epsilon=0.00001, reuse=False, id=''):
        super().__init__(graph, var2std_epsilon, reuse, id)
        self.name = 'xvector'
        self.id = self.get_version_id()

    def __get_variable(self, name, shape, initializer, trainable=True):
        return tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32, trainable=trainable)

    def batch_norm_wrapper(self, inputs, is_training, decay=0.99, epsilon=1e-3, name_prefix=''):
        gamma = self.__get_variable(name_prefix + 'gamma', inputs.get_shape()[-1], tf.constant_initializer(1.0))
        beta = self.__get_variable(name_prefix + 'beta', inputs.get_shape()[-1], tf.constant_initializer(0.0))
        pop_mean = self.__get_variable(name_prefix + 'mean', inputs.get_shape()[-1], tf.constant_initializer(0.0), trainable=False)
        pop_var = self.__get_variable(name_prefix + 'variance', inputs.get_shape()[-1], tf.constant_initializer(1.0), trainable=False)
        axis = list(range(len(inputs.get_shape()) - 1))

        def in_training():
            batch_mean, batch_var = tf.nn.moments(inputs, axis)
            train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
            train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, gamma, epsilon)

        def in_evaluation():
            return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, gamma, epsilon)

        return tf.cond(is_training, lambda: in_training(), lambda: in_evaluation())

    def build(self, input_x, input_y, n_classes=0, n_filters=24, noises=None, cache=None, augment=0, n_seconds=3, sample_rate=16000):

        with self.graph.as_default():
            print('>', 'building', self.name, 'model')

            super().build(input_x, input_y, n_classes, n_filters, noises, cache, augment, n_seconds, sample_rate)
            self.input_s = tf.identity(get_tf_mfccs(self.input_a, self.sample_rate, self.frame_size, self.frame_stride, self.num_fft, self.n_filters, self.lower_edge_hertz, self.upper_edge_hertz), name='input_s')

            layer_sizes = [512, 512, 512, 512, 3 * 512]
            kernel_sizes = [5, 5, 7, 1, 1]
            embedding_sizes = [512, 512]
            h = self.input_s

            # Frame level information Layer
            prev_dim = n_filters
            for i, (kernel_size, layer_size) in enumerate(zip(kernel_sizes, layer_sizes)):
                with tf.variable_scope("frame_level_info_layer-%s" % i):
                    kernel_shape = [kernel_size, prev_dim, layer_size]
                    w = tf.Variable(tf.truncated_normal(kernel_shape, stddev=0.1), name="w")
                    b = tf.Variable(tf.constant(0.1, shape=[layer_size]), name="b")
                    conv = tf.nn.conv1d(h, w, stride=1, padding="SAME", name="conv-layer-%s" % i)
                    h = tf.nn.bias_add(conv, b)

                    # Apply nonlinearity and BN
                    h = tf.nn.relu(h, name="relu")
                    h = self.batch_norm_wrapper(h, decay=0.95, is_training=self.phase)
                    prev_dim = layer_size

                    # Apply dropout
                    if i != len(kernel_sizes) - 1:
                        with tf.name_scope("dropout-%s" % i):
                            h = tf.nn.dropout(h, self.dropout_keep_prob)

            # Statistic pooling
            tf_mean, tf_var = tf.nn.moments(h, 1)
            h = tf.concat([tf_mean, tf.sqrt(tf_var + self.var2std_epsilon)], 1)
            prev_dim = prev_dim * 2

            # Embedding layers
            for i, out_dim in enumerate(embedding_sizes):

                with tf.variable_scope("embed_layer-%s" % i):
                    w = tf.Variable(tf.truncated_normal([prev_dim, out_dim], stddev=0.1), name="w")
                    b = tf.Variable(tf.constant(0.1, shape=[out_dim]), name="b")
                    h = tf.nn.xw_plus_b(h, w, b, name="scores")
                    h = tf.nn.relu(h, name="relu")
                    h = self.batch_norm_wrapper(h, decay=0.95, is_training=self.phase)

                    prev_dim = out_dim
                    if i != len(embedding_sizes) - 1:
                        with tf.name_scope("dropout-%s" % i):
                            h = tf.nn.dropout(h, self.dropout_keep_prob)

            self.embedding = self.graph.get_tensor_by_name('embed_layer-0/scores:0')[0]

            # Softmax
            with tf.variable_scope("output"):
                w = tf.get_variable("w", shape=[prev_dim, n_classes], initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.constant(0.1, shape=[n_classes]), name="b")
                scores = tf.nn.xw_plus_b(h, w, b, name="scores")
                predictions = tf.argmax(scores, 1, name="predictions", output_type=tf.int32)

            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses, name="loss")

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss, name="optimizer")

            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(predictions, self.input_y)
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')

        if not self.reuse:
            with tf.Session(graph=self.graph, config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
                print('>', 'initializing', self.name, 'graph')
                sess.run(tf.global_variables_initializer())
                self.save(sess)

        print('>', self.name, 'built finished')

