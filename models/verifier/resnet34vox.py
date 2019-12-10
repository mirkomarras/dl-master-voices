#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import queue
import time
import os
import random

from models.verifier.model import Model
from helpers.audio import get_tf_spectrum

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class ResNet34Vox(Model):

    def __init__(self, graph=tf.Graph(), var2std_epsilon=0.00001, reuse=False, id=''):
        super().__init__(graph, var2std_epsilon, reuse, id)
        self.name = 'resnet34vox'
        self.id = self.get_version_id()

    def identity_block2d(self, input_tensor, kernel_size, filters, stage, block, is_training, reuse, kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False)):
        filters1, filters2, filters3 = filters

        conv_name_1 = 'conv' + str(stage) + '_' + str(block) + '_1x1_reduce'
        bn_name_1 = 'bn' + str(stage) + '_' + str(block) + '_1x1_reduce'

        x = tf.layers.conv2d(input_tensor, filters1, (1, 1), use_bias=False, kernel_initializer=kernel_initializer, name=conv_name_1, reuse=reuse)
        x = tf.layers.batch_normalization(x, training=is_training, name=bn_name_1, reuse=reuse)
        x = tf.nn.relu(x)

        conv_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3'
        bn_name_2 = 'bn' + str(stage) + '_' + str(block) + '_3x3'
        x = tf.layers.conv2d(x, filters2, kernel_size, use_bias=False, padding='SAME', kernel_initializer=kernel_initializer, name=conv_name_2, reuse=reuse)
        x = tf.layers.batch_normalization(x, training=is_training, name=bn_name_2, reuse=reuse)
        x = tf.nn.relu(x)

        conv_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase'
        bn_name_3 = 'bn' + str(stage) + '_' + str(block) + '_1x1_increase'
        x = tf.layers.conv2d(x, filters3, (1, 1), use_bias=False, kernel_initializer=kernel_initializer, name=conv_name_3, reuse=reuse)
        x = tf.layers.batch_normalization(x, training=is_training, name=bn_name_3, reuse=reuse)

        x = tf.add(input_tensor, x)
        x = tf.nn.relu(x)
        return x

    def conv_block_2d(self, input_tensor, kernel_size, filters, stage, block, is_training, reuse, strides=(2, 2), kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False)):
        filters1, filters2, filters3 = filters

        conv_name_1 = 'conv' + str(stage) + '_' + str(block) + '_1x1_reduce'
        bn_name_1 = 'bn' + str(stage) + '_' + str(block) + '_1x1_reduce'
        x = tf.layers.conv2d(input_tensor, filters1, (1, 1), strides=strides, use_bias=False, kernel_initializer=kernel_initializer, name=conv_name_1, reuse=reuse)
        x = tf.layers.batch_normalization(x, training=is_training, name=bn_name_1, reuse=reuse)
        x = tf.nn.relu(x)

        conv_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3'
        bn_name_2 = 'bn' + str(stage) + '_' + str(block) + '_3x3'
        x = tf.layers.conv2d(x, filters2, kernel_size, padding='SAME', use_bias=False, kernel_initializer=kernel_initializer, name=conv_name_2, reuse=reuse)
        x = tf.layers.batch_normalization(x, training=is_training, name=bn_name_2, reuse=reuse)
        x = tf.nn.relu(x)

        conv_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase'
        bn_name_3 = 'bn' + str(stage) + '_' + str(block) + '_1x1_increase'
        x = tf.layers.conv2d(x, filters3, (1, 1), use_bias=False, kernel_initializer=kernel_initializer, name=conv_name_3, reuse=reuse)
        x = tf.layers.batch_normalization(x, training=is_training, name=bn_name_3, reuse=reuse)

        conv_name_4 = 'conv' + str(stage) + '_' + str(block) + '_1x1_shortcut'
        bn_name_4 = 'bn' + str(stage) + '_' + str(block) + '_1x1_shortcut'
        shortcut = tf.layers.conv2d(input_tensor, filters3, (1, 1), use_bias=False, strides=strides, kernel_initializer=kernel_initializer, name=conv_name_4, reuse=reuse)
        shortcut = tf.layers.batch_normalization(shortcut, training=is_training, name=bn_name_4, reuse=reuse)

        x = tf.add(shortcut, x)
        x = tf.nn.relu(x)
        return x

    def build(self, input_x, input_y, n_classes=0, n_filters=24, noises=None, cache=None, augment=0, n_seconds=3, sample_rate=16000):

        with self.graph.as_default():
            print('>', 'building', self.name, 'model')

            super().build(input_x, input_y, n_classes, n_filters, noises, cache, augment, n_seconds, sample_rate)
            self.input_s = tf.identity(get_tf_spectrum(self.input_a, self.sample_rate, self.frame_size, self.frame_stride, self.num_fft), name='input_s')

            kernel_initializer = tf.contrib.layers.xavier_initializer(uniform=False)
            input_x = tf.layers.batch_normalization(self.input_s, training=self.phase, name='bbn0', reuse=self.reuse)

            x = tf.layers.conv2d(input_x, 64, (7, 7), strides=(1, 1), kernel_initializer=kernel_initializer, use_bias=False, padding='SAME', name='voice_conv1_1/3x3_s1', reuse=self.reuse)
            x = tf.layers.batch_normalization(x, training=self.phase, name='voice_bn1_1/3x3_s1', reuse=self.reuse)
            x = tf.nn.relu(x)
            x = tf.layers.max_pooling2d(x, (2, 2), strides=(2, 2), name='voice_mpool1')

            x1 = self.conv_block_2d(x, 3, [48, 48, 96], stage=2, block='voice_1a', strides=(1, 1), is_training=self.phase, reuse=self.reuse, kernel_initializer=kernel_initializer)
            x1 = self.identity_block2d(x1, 3, [48, 48, 96], stage=2, block='voice_1b', is_training=self.phase, reuse=self.reuse, kernel_initializer=kernel_initializer)

            x2 = self.conv_block_2d(x1, 3, [96, 96, 128], stage=3, block='voice_2a', strides=(2, 2), is_training=self.phase, reuse=self.reuse, kernel_initializer=kernel_initializer)
            x2 = self.identity_block2d(x2, 3, [96, 96, 128], stage=3, block='voice_2b', is_training=self.phase, reuse=self.reuse, kernel_initializer=kernel_initializer)
            x2 = self.identity_block2d(x2, 3, [96, 96, 128], stage=3, block='voice_2c', is_training=self.phase, reuse=self.reuse, kernel_initializer=kernel_initializer)

            x3 = self.conv_block_2d(x2, 3, [128, 128, 256], stage=4, block='voice_3a', strides=(2, 2), is_training=self.phase, reuse=self.reuse, kernel_initializer=kernel_initializer)
            x3 = self.identity_block2d(x3, 3, [128, 128, 256], stage=4, block='voice_3b', is_training=self.phase, reuse=self.reuse, kernel_initializer=kernel_initializer)
            x3 = self.identity_block2d(x3, 3, [128, 128, 256], stage=4, block='voice_3c', is_training=self.phase, reuse=self.reuse, kernel_initializer=kernel_initializer)

            x4 = self.conv_block_2d(x3, 3, [256, 256, 512], stage=5, block='voice_4a', is_training=self.phase, reuse=self.reuse, kernel_initializer=kernel_initializer)
            x4 = self.identity_block2d(x4, 3, [256, 256, 512], stage=5, block='voice_4b', is_training=self.phase, reuse=self.reuse, kernel_initializer=kernel_initializer)
            x4 = self.identity_block2d(x4, 3, [256, 256, 512], stage=5, block='voice_4c', is_training=self.phase, reuse=self.reuse, kernel_initializer=kernel_initializer)

            with tf.variable_scope('fc'):
                pooling_output = tf.layers.max_pooling2d(x4, (3, 1), strides=(2, 2), name='voice_mpool2')
                pooling_output = tf.layers.conv2d(pooling_output, filters=512, kernel_size=[7, 1], strides=[1, 1], padding='SAME', activation=tf.nn.relu, name='fc_block1')
                fc1 = tf.layers.conv2d(pooling_output, filters=512, kernel_size=[7, 1], strides=[1, 1], padding='SAME', activation=tf.nn.relu, name='fc_block1_conv')
                pooling_output = tf.reduce_mean(fc1, [1, 2], name='gap')
                fc2 = tf.layers.dense(pooling_output, 512, activation=tf.nn.relu, name='fc2')

                flattened = tf.contrib.layers.flatten(fc2)
                flattened = tf.nn.l2_normalize(flattened)
                w = tf.Variable(tf.truncated_normal([512, 512], stddev=0.1), name="w")
                b = tf.Variable(tf.constant(0.1, shape=[512]), name="b")

                h = tf.nn.xw_plus_b(flattened, w, b, name="scores")
                h = tf.nn.relu(h, name="relu")

                with tf.name_scope("dropout"):
                    h = tf.nn.dropout(h, self.dropout_keep_prob)

            self.embedding = self.graph.get_tensor_by_name('fc/scores:0')[0]

            with tf.variable_scope('output'):
                w = tf.get_variable('w', shape=[512, n_classes], initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.constant(0.1, shape=[n_classes]), name="b")
                scores = tf.nn.xw_plus_b(h, w, b, name="scores")
                predictions = tf.argmax(scores, 1, name="predictions", output_type=tf.int32)

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
