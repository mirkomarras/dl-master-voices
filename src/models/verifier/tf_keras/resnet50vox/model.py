#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import queue
import random
import time

import numpy as np
import tensorflow as tf
from src.helpers.audio import get_tf_spectrum
from src.models.verifier.tf.model import Model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class ResNet50Vox(Model):

    def __init__(self, graph=tf.Graph(), var2std_epsilon=0.00001, reuse=False, id=''):
        super().__init__(graph, var2std_epsilon, reuse, id)
        self.name = 'resnet50vox'
        self.id = self.get_version_id()

    def build(self, input_x, input_y, n_classes=0, n_filters=24, noises=None, cache=None, augment=0, n_seconds=3, sample_rate=16000):
        with self.graph.as_default():
            print('>', 'building', self.name, 'model')

            super().build(input_x, input_y, n_classes, n_filters, noises, cache, augment, n_seconds, sample_rate)
            self.input_s = tf.identity(get_tf_spectrum(self.input_a, self.sample_rate, self.frame_size, self.frame_stride, self.num_fft), name='input_s')

            with tf.name_scope('block0'):
                conv0_1 = tf.layers.conv2d(self.input_s, filters=64, kernel_size=[7, 7], strides=[2, 2], padding='SAME', reuse=self.reuse, name='conv1')
                conv0_1 = tf.layers.batch_normalization(conv0_1, training=self.phase, name='bbn0', reuse=self.reuse)
                conv0_1 = tf.nn.relu(conv0_1)
                conv0_1 = tf.layers.max_pooling2d(conv0_1, pool_size=[3, 3], strides=[2, 2], name='mpool1')

            with tf.name_scope('block1'):
                with tf.variable_scope('block1_conv1') as scope:
                    conv_block1_conv1_shortcut = tf.layers.conv2d(conv0_1, filters=256, kernel_size=[1, 1], strides=[1, 1], padding='VALID', reuse=self.reuse, name='conv_block1_conv1_shortcut_conv')
                    conv_block1_conv1_shortcut = tf.layers.batch_normalization(conv_block1_conv1_shortcut, training=self.phase, name='conv_block1_conv1_shortcut_bn', reuse=self.reuse)

                    conv_block1_conv1_1 = tf.layers.conv2d(conv0_1, filters=64, kernel_size=[1, 1], strides=[1, 1], padding='VALID', reuse=self.reuse, name='conv_block1_conv1_1')
                    conv_block1_conv1_1 = tf.layers.batch_normalization(conv_block1_conv1_1, training=self.phase, name='conv_block1_conv1_1_bn', reuse=self.reuse)
                    conv_block1_conv1_1 = tf.nn.relu(conv_block1_conv1_1)

                    conv_block1_conv1_2 = tf.layers.conv2d(conv_block1_conv1_1, filters=64, kernel_size=[3, 3], strides=[1, 1], padding='SAME', reuse=self.reuse, name='conv_block1_conv1_2')
                    conv_block1_conv1_2 = tf.layers.batch_normalization(conv_block1_conv1_2, training=self.phase, name='conv_block1_conv1_2_bn', reuse=self.reuse)
                    conv_block1_conv1_2 = tf.nn.relu(conv_block1_conv1_2)

                    conv_block1_conv1_3 = tf.layers.conv2d(conv_block1_conv1_2, filters=256, kernel_size=[1, 1], strides=[1, 1], padding='VALID', reuse=self.reuse, name='conv_block1_conv1_3')
                    conv_block1_conv1_3 = tf.layers.batch_normalization(conv_block1_conv1_3, training=self.phase, name='conv_block1_conv1_3_bn', reuse=self.reuse)
                    # conv_block1_conv1_3 = tf.nn.relu(conv_block1_conv1_3)
                    conv_block1_output1 = conv_block1_conv1_shortcut + conv_block1_conv1_3
                    conv_block1_output1 = tf.nn.relu(conv_block1_output1)

                with tf.variable_scope('block1_conv2') as scope:
                    # conv_block1_conv2_shortcut = conv_block1_output1
                    # conv_block1_conv2_shortcut = tf.layers.batch_normalization(conv_block1_conv1_shortcut, training=self.phase, name='conv_block1_conv1_shortcut_bn', reuse=self.reuse)

                    conv_block1_conv2_1 = tf.layers.conv2d(conv_block1_output1, filters=64, kernel_size=[1, 1], strides=[1, 1], padding='SAME', reuse=self.reuse, name='conv_block1_conv2_1')
                    conv_block1_conv2_1 = tf.layers.batch_normalization(conv_block1_conv2_1, training=self.phase, name='conv_block1_conv2_1_bn', reuse=self.reuse)
                    conv_block1_conv2_1 = tf.nn.relu(conv_block1_conv2_1)

                    conv_block1_conv2_2 = tf.layers.conv2d(conv_block1_conv2_1, filters=64, kernel_size=[3, 3], strides=[1, 1], padding='SAME', reuse=self.reuse, name='conv_block1_conv2_2')
                    conv_block1_conv2_2 = tf.layers.batch_normalization(conv_block1_conv2_2, training=self.phase, name='conv_block1_conv2_2_bn', reuse=self.reuse)
                    conv_block1_conv2_2 = tf.nn.relu(conv_block1_conv2_2)

                    conv_block1_conv2_3 = tf.layers.conv2d(conv_block1_conv2_2, filters=256, kernel_size=[1, 1], strides=[1, 1], padding='SAME', reuse=self.reuse, name='conv_block1_conv2_3')
                    conv_block1_conv2_3 = tf.layers.batch_normalization(conv_block1_conv2_3, training=self.phase, name='conv_block1_conv2_3_bn', reuse=self.reuse)
                    # conv_block1_conv1_3 = tf.nn.relu(conv_block1_conv1_3)
                    conv_block1_output2 = conv_block1_output1 + conv_block1_conv2_3
                    conv_block1_output2 = tf.nn.relu(conv_block1_output2)

                with tf.variable_scope('block1_conv3') as scope:
                    # conv_block1_conv2_shortcut = conv_block1_output1
                    # conv_block1_conv2_shortcut = tf.layers.batch_normalization(conv_block1_conv1_shortcut, training=self.phase, name='conv_block1_conv1_shortcut_bn', reuse=self.reuse)

                    conv_block1_conv3_1 = tf.layers.conv2d(conv_block1_output2, filters=64, kernel_size=[1, 1], strides=[1, 1], padding='SAME', reuse=self.reuse, name='conv_block1_conv3_1')
                    conv_block1_conv3_1 = tf.layers.batch_normalization(conv_block1_conv3_1, training=self.phase, name='conv_block1_conv3_1_bn', reuse=self.reuse)
                    conv_block1_conv3_1 = tf.nn.relu(conv_block1_conv3_1)

                    conv_block1_conv3_2 = tf.layers.conv2d(conv_block1_conv3_1, filters=64, kernel_size=[3, 3], strides=[1, 1], padding='SAME', reuse=self.reuse, name='conv_block1_conv3_2')
                    conv_block1_conv3_2 = tf.layers.batch_normalization(conv_block1_conv3_2, training=self.phase, name='conv_block1_conv3_2_bn', reuse=self.reuse)
                    conv_block1_conv3_2 = tf.nn.relu(conv_block1_conv3_2)

                    conv_block1_conv3_3 = tf.layers.conv2d(conv_block1_conv3_2, filters=256, kernel_size=[1, 1], strides=[1, 1], padding='SAME', reuse=self.reuse, name='conv_block1_conv3_3')
                    conv_block1_conv3_3 = tf.layers.batch_normalization(conv_block1_conv3_3, training=self.phase, name='cconv_block1_conv3_3_bn', reuse=self.reuse)
                    # conv_block1_conv1_3 = tf.nn.relu(conv_block1_conv1_3)
                    conv_block1_output3 = conv_block1_output2 + conv_block1_conv3_3
                    conv_block1_output3 = tf.nn.relu(conv_block1_output2)

            with tf.name_scope('block2'):
                with tf.variable_scope('block2_conv1') as scope:
                    conv_block2_conv1_shortcut = tf.layers.conv2d(conv_block1_output3, filters=512, kernel_size=[1, 1], strides=[2, 2], padding='SAME', reuse=self.reuse, name='conv_block2_conv1_shortcut_conv')
                    conv_block2_conv1_shortcut = tf.layers.batch_normalization(conv_block2_conv1_shortcut, training=self.phase, name='conv_block2_conv1_shortcut_bn', reuse=self.reuse)
                    conv_block2_conv1_1 = tf.layers.conv2d(conv_block1_output3, filters=128, kernel_size=[1, 1], strides=[2, 2], padding='SAME', reuse=self.reuse, name='conv_block2_conv1_1')
                    conv_block2_conv1_1 = tf.layers.batch_normalization(conv_block2_conv1_1, training=self.phase, name='conv_block2_conv1_1_bn', reuse=self.reuse)
                    conv_block2_conv1_1 = tf.nn.relu(conv_block2_conv1_1)

                    conv_block2_conv1_2 = tf.layers.conv2d(conv_block2_conv1_1, filters=128, kernel_size=[3, 3], strides=[1, 1], padding='SAME', reuse=self.reuse, name='conv_block2_conv1_2')
                    conv_block2_conv1_2 = tf.layers.batch_normalization(conv_block2_conv1_2, training=self.phase, name='conv_block2_conv1_2_bn', reuse=self.reuse)
                    conv_block2_conv1_2 = tf.nn.relu(conv_block2_conv1_2)

                    conv_block2_conv1_3 = tf.layers.conv2d(conv_block2_conv1_2, filters=512, kernel_size=[1, 1], strides=[1, 1], padding='SAME', reuse=self.reuse, name='conv_block2_conv1_3')
                    conv_block2_conv1_3 = tf.layers.batch_normalization(conv_block2_conv1_3, training=self.phase, name='conv_block2_conv1_3_bn', reuse=self.reuse)
                    # conv_block1_conv1_3 = tf.nn.relu(conv_block1_conv1_3)
                    conv_block2_output1 = conv_block2_conv1_shortcut + conv_block2_conv1_3
                    conv_block2_output1 = tf.nn.relu(conv_block2_output1)

                with tf.variable_scope('block2_conv2') as scope:
                    # conv_block2_conv1_shortcut = tf.layers.conv2d(conv_block1_output3, filters=512, kernel_size=[1,1], strides=[1,1], padding='SAME', reuse=self.reuse, name='conv_block2_conv1_shortcut_conv')
                    # conv_block2_conv1_shortcut = tf.layers.batch_normalization(conv_block2_conv1_shortcut, training=self.phase, name='conv_block2_conv1_shortcut_bn', reuse=self.reuse)

                    conv_block2_conv2_1 = tf.layers.conv2d(conv_block2_output1, filters=128, kernel_size=[1, 1], strides=[1, 1], padding='SAME', reuse=self.reuse, name='conv_block2_conv2_1')
                    conv_block2_conv2_1 = tf.layers.batch_normalization(conv_block2_conv2_1, training=self.phase, name='conv_block2_conv2_1_bn', reuse=self.reuse)
                    conv_block2_conv2_1 = tf.nn.relu(conv_block2_conv2_1)

                    conv_block2_conv2_2 = tf.layers.conv2d(conv_block2_conv2_1, filters=128, kernel_size=[3, 3], strides=[1, 1], padding='SAME', reuse=self.reuse, name='conv_block2_conv2_2')
                    conv_block2_conv2_2 = tf.layers.batch_normalization(conv_block2_conv2_2, training=self.phase, name='conv_block2_conv2_2_bn', reuse=self.reuse)
                    conv_block2_conv2_2 = tf.nn.relu(conv_block2_conv2_2)

                    conv_block2_conv2_3 = tf.layers.conv2d(conv_block2_conv2_2, filters=512, kernel_size=[1, 1], strides=[1, 1], padding='SAME', reuse=self.reuse, name='conv_block2_conv2_3')
                    conv_block2_conv2_3 = tf.layers.batch_normalization(conv_block2_conv2_3, training=self.phase, name='conv_block2_conv2_3_bn', reuse=self.reuse)
                    # conv_block1_conv1_3 = tf.nn.relu(conv_block1_conv1_3)
                    conv_block2_output2 = conv_block2_output1 + conv_block2_conv2_3
                    conv_block2_output2 = tf.nn.relu(conv_block2_output2)

                with tf.variable_scope('block2_conv3') as scope:
                    # conv_block2_conv1_shortcut = tf.layers.conv2d(conv_block1_output3, filters=512, kernel_size=[1,1], strides=[1,1], padding='SAME', reuse=self.reuse, name='conv_block2_conv1_shortcut_conv')
                    # conv_block2_conv1_shortcut = tf.layers.batch_normalization(conv_block2_conv1_shortcut, training=self.phase, name='conv_block2_conv1_shortcut_bn', reuse=self.reuse)

                    conv_block2_conv3_1 = tf.layers.conv2d(conv_block2_output2, filters=128, kernel_size=[1, 1], strides=[1, 1], padding='SAME', reuse=self.reuse, name='conv_block2_conv3_1')
                    conv_block2_conv3_1 = tf.layers.batch_normalization(conv_block2_conv3_1, training=self.phase, name='conv_block2_conv3_1_bn', reuse=self.reuse)
                    conv_block2_conv3_1 = tf.nn.relu(conv_block2_conv3_1)

                    conv_block2_conv3_2 = tf.layers.conv2d(conv_block2_conv3_1, filters=128, kernel_size=[3, 3], strides=[1, 1], padding='SAME', reuse=self.reuse, name='conv_block2_conv3_2')
                    conv_block2_conv3_2 = tf.layers.batch_normalization(conv_block2_conv3_2, training=self.phase, name='conv_block2_conv3_2_bn', reuse=self.reuse)
                    conv_block2_conv3_2 = tf.nn.relu(conv_block2_conv3_2)

                    conv_block2_conv3_3 = tf.layers.conv2d(conv_block2_conv3_2, filters=512, kernel_size=[1, 1], strides=[1, 1], padding='SAME', reuse=self.reuse, name='conv_block2_conv3_3')
                    conv_block2_conv3_3 = tf.layers.batch_normalization(conv_block2_conv3_3, training=self.phase, name='conv_block2_conv3_3_bn', reuse=self.reuse)
                    # conv_block1_conv1_3 = tf.nn.relu(conv_block1_conv1_3)
                    conv_block2_output3 = conv_block2_output2 + conv_block2_conv3_3
                    conv_block2_output3 = tf.nn.relu(conv_block2_output3)

                with tf.variable_scope('block2_conv4') as scope:
                    # conv_block2_conv1_shortcut = tf.layers.conv2d(conv_block1_output3, filters=512, kernel_size=[1,1], strides=[1,1], padding='SAME', reuse=self.reuse, name='conv_block2_conv1_shortcut_conv')
                    # conv_block2_conv1_shortcut = tf.layers.batch_normalization(conv_block2_conv1_shortcut, training=self.phase, name='conv_block2_conv1_shortcut_bn', reuse=self.reuse)

                    conv_block2_conv4_1 = tf.layers.conv2d(conv_block2_output3, filters=128, kernel_size=[1, 1], strides=[1, 1], padding='SAME', reuse=self.reuse, name='conv_block2_conv4_1')
                    conv_block2_conv4_1 = tf.layers.batch_normalization(conv_block2_conv4_1, training=self.phase, name='conv_block2_conv4_1_bn', reuse=self.reuse)
                    conv_block2_conv4_1 = tf.nn.relu(conv_block2_conv4_1)

                    conv_block2_conv4_2 = tf.layers.conv2d(conv_block2_conv4_1, filters=128, kernel_size=[3, 3], strides=[1, 1], padding='SAME', reuse=self.reuse, name='conv_block2_conv4_2')
                    conv_block2_conv4_2 = tf.layers.batch_normalization(conv_block2_conv4_2, training=self.phase, name='conv_block2_conv4_2_bn', reuse=self.reuse)
                    conv_block2_conv4_2 = tf.nn.relu(conv_block2_conv4_2)

                    conv_block2_conv4_3 = tf.layers.conv2d(conv_block2_conv4_2, filters=512, kernel_size=[1, 1], strides=[1, 1], padding='SAME', reuse=self.reuse, name='conv_block2_conv4_3')
                    conv_block2_conv4_3 = tf.layers.batch_normalization(conv_block2_conv4_3, training=self.phase, name='conv_block2_conv4_3_bn', reuse=self.reuse)
                    # conv_block1_conv1_3 = tf.nn.relu(conv_block1_conv1_3)
                    conv_block2_output4 = conv_block2_output3 + conv_block2_conv4_3
                    conv_block2_output4 = tf.nn.relu(conv_block2_output4)

            with tf.name_scope('block3'):
                with tf.variable_scope('block3_conv1') as scope:
                    conv_block3_conv1_shortcut = tf.layers.conv2d(conv_block2_output4, filters=1024, kernel_size=[1, 1], strides=[2, 2], padding='SAME', reuse=self.reuse, name='conv_block3_conv1_shortcut')
                    conv_block3_conv1_shortcut = tf.layers.batch_normalization(conv_block3_conv1_shortcut, training=self.phase, name='conv_block3_conv1_shortcut_bn', reuse=self.reuse)

                    conv_block3_conv1_1 = tf.layers.conv2d(conv_block2_output4, filters=256, kernel_size=[1, 1], strides=[2, 2], padding='SAME', reuse=self.reuse, name='conv_block3_conv1_1')
                    conv_block3_conv1_1 = tf.layers.batch_normalization(conv_block3_conv1_1, training=self.phase, name='conv_block3_conv1_1_bn', reuse=self.reuse)
                    conv_block3_conv1_1 = tf.nn.relu(conv_block3_conv1_1)

                    conv_block3_conv1_2 = tf.layers.conv2d(conv_block3_conv1_1, filters=256, kernel_size=[3, 3], strides=[1, 1], padding='SAME', reuse=self.reuse, name='conv_block3_conv1_2')
                    conv_block3_conv1_2 = tf.layers.batch_normalization(conv_block3_conv1_2, training=self.phase, name='conv_block3_conv1_2_bn', reuse=self.reuse)
                    conv_block3_conv1_2 = tf.nn.relu(conv_block3_conv1_2)

                    conv_block3_conv1_3 = tf.layers.conv2d(conv_block3_conv1_2, filters=1024, kernel_size=[1, 1], strides=[1, 1], padding='SAME', reuse=self.reuse, name='conv_block3_conv1_3')
                    conv_block3_conv1_3 = tf.layers.batch_normalization(conv_block3_conv1_3, training=self.phase, name='conv_block3_conv1_3_bn', reuse=self.reuse)
                    # conv_block1_conv1_3 = tf.nn.relu(conv_block1_conv1_3)
                    conv_block3_output1 = conv_block3_conv1_shortcut + conv_block3_conv1_3
                    conv_block3_output1 = tf.nn.relu(conv_block3_output1)

                with tf.variable_scope('block3_conv2') as scope:
                    # conv_block3_conv1_shortcut = tf.layers.conv2d(conv_block2_output4, filters=1024, kernel_size=[1,1], strides=[1,1], padding='SAME', reuse=self.reuse, name='conv_block3_conv1_shortcut')
                    # conv_block3_conv1_shortcut = tf.layers.batch_normalization(conv_block3_conv1_shortcut, training=self.phase, name='conv_block3_conv1_shortcut_bn', reuse=self.reuse)

                    conv_block3_conv2_1 = tf.layers.conv2d(conv_block3_output1, filters=256, kernel_size=[1, 1], strides=[1, 1], padding='SAME', reuse=self.reuse, name='conv_block3_conv2_1')
                    conv_block3_conv2_1 = tf.layers.batch_normalization(conv_block3_conv2_1, training=self.phase, name='conv_block3_conv2_1_bn', reuse=self.reuse)
                    conv_block3_conv2_1 = tf.nn.relu(conv_block3_conv2_1)

                    conv_block3_conv2_2 = tf.layers.conv2d(conv_block3_conv2_1, filters=256, kernel_size=[3, 3], strides=[1, 1], padding='SAME', reuse=self.reuse, name='conv_block3_conv2_2')
                    conv_block3_conv2_2 = tf.layers.batch_normalization(conv_block3_conv2_2, training=self.phase, name='conv_block3_conv2_2_bn', reuse=self.reuse)
                    conv_block3_conv2_2 = tf.nn.relu(conv_block3_conv2_2)

                    conv_block3_conv2_3 = tf.layers.conv2d(conv_block3_conv2_2, filters=1024, kernel_size=[1, 1], strides=[1, 1], padding='SAME', reuse=self.reuse, name='conv_block3_conv2_3')
                    conv_block3_conv2_3 = tf.layers.batch_normalization(conv_block3_conv1_3, training=self.phase, name='conv_block3_conv2_3_bn', reuse=self.reuse)
                    # conv_block1_conv1_3 = tf.nn.relu(conv_block1_conv1_3)
                    conv_block3_output2 = conv_block3_output1 + conv_block3_conv2_3
                    conv_block3_output2 = tf.nn.relu(conv_block3_output2)

                with tf.variable_scope('block3_conv3') as scope:
                    # conv_block3_conv1_shortcut = tf.layers.conv2d(conv_block2_output4, filters=1024, kernel_size=[1,1], strides=[1,1], padding='SAME', reuse=self.reuse, name='conv_block3_conv1_shortcut')
                    # conv_block3_conv1_shortcut = tf.layers.batch_normalization(conv_block3_conv1_shortcut, training=self.phase, name='conv_block3_conv1_shortcut_bn', reuse=self.reuse)

                    conv_block3_conv3_1 = tf.layers.conv2d(conv_block3_output2, filters=256, kernel_size=[1, 1], strides=[1, 1], padding='SAME', reuse=self.reuse, name='conv_block3_conv3_1')
                    conv_block3_conv3_1 = tf.layers.batch_normalization(conv_block3_conv3_1, training=self.phase, name='conv_block3_conv3_1_1_bn', reuse=self.reuse)
                    conv_block3_conv3_1 = tf.nn.relu(conv_block3_conv3_1)

                    conv_block3_conv3_2 = tf.layers.conv2d(conv_block3_conv3_1, filters=256, kernel_size=[3, 3], strides=[1, 1], padding='SAME', reuse=self.reuse, name='conv_block3_conv3_2')
                    conv_block3_conv3_2 = tf.layers.batch_normalization(conv_block3_conv3_2, training=self.phase, name='conv_block3_conv3_2_bn', reuse=self.reuse)
                    conv_block3_conv3_2 = tf.nn.relu(conv_block3_conv3_2)

                    conv_block3_conv3_3 = tf.layers.conv2d(conv_block3_conv3_2, filters=1024, kernel_size=[1, 1], strides=[1, 1], padding='SAME', reuse=self.reuse, name='conv_block3_conv3_3')
                    conv_block3_conv3_3 = tf.layers.batch_normalization(conv_block3_conv3_3, training=self.phase, name='conv_block3_conv3_3_bn', reuse=self.reuse)
                    # conv_block1_conv1_3 = tf.nn.relu(conv_block1_conv1_3)
                    conv_block3_output3 = conv_block3_output2 + conv_block3_conv3_3
                    conv_block3_output3 = tf.nn.relu(conv_block3_output3)

                with tf.variable_scope('block3_conv4') as scope:
                    # conv_block3_conv1_shortcut = tf.layers.conv2d(conv_block2_output4, filters=1024, kernel_size=[1,1], strides=[1,1], padding='SAME', reuse=self.reuse, name='conv_block3_conv1_shortcut')
                    # conv_block3_conv1_shortcut = tf.layers.batch_normalization(conv_block3_conv1_shortcut, training=self.phase, name='conv_block3_conv1_shortcut_bn', reuse=self.reuse)

                    conv_block3_conv4_1 = tf.layers.conv2d(conv_block3_output3, filters=256, kernel_size=[1, 1], strides=[1, 1], padding='SAME', reuse=self.reuse, name='conv_block3_conv4_1')
                    conv_block3_conv4_1 = tf.layers.batch_normalization(conv_block3_conv4_1, training=self.phase, name='conv_block3_conv4_1_bn', reuse=self.reuse)
                    conv_block3_conv4_1 = tf.nn.relu(conv_block3_conv4_1)

                    conv_block3_conv4_2 = tf.layers.conv2d(conv_block3_conv4_1, filters=256, kernel_size=[3, 3], strides=[1, 1], padding='SAME', reuse=self.reuse, name='conv_block3_conv4_2')
                    conv_block3_conv4_2 = tf.layers.batch_normalization(conv_block3_conv4_2, training=self.phase, name='conv_block3_conv4_2_bn', reuse=self.reuse)
                    conv_block3_conv4_2 = tf.nn.relu(conv_block3_conv4_2)

                    conv_block3_conv4_3 = tf.layers.conv2d(conv_block3_conv4_2, filters=1024, kernel_size=[1, 1], strides=[1, 1], padding='SAME', reuse=self.reuse, name='conv_block3_conv4_3')
                    conv_block3_conv4_3 = tf.layers.batch_normalization(conv_block3_conv3_3, training=self.phase, name='conv_block3_conv4_3_bn', reuse=self.reuse)
                    # conv_block1_conv1_3 = tf.nn.relu(conv_block1_conv1_3)
                    conv_block3_output4 = conv_block3_output3 + conv_block3_conv4_3
                    conv_block3_output4 = tf.nn.relu(conv_block3_output4)

                with tf.variable_scope('block3_conv5') as scope:
                    # conv_block3_conv1_shortcut = tf.layers.conv2d(conv_block2_output4, filters=1024, kernel_size=[1,1], strides=[1,1], padding='SAME', reuse=self.reuse, name='conv_block3_conv1_shortcut')
                    # conv_block3_conv1_shortcut = tf.layers.batch_normalization(conv_block3_conv1_shortcut, training=self.phase, name='conv_block3_conv1_shortcut_bn', reuse=self.reuse)

                    conv_block3_conv5_1 = tf.layers.conv2d(conv_block3_output4, filters=256, kernel_size=[1, 1], strides=[1, 1], padding='SAME', reuse=self.reuse, name='conv_block3_conv5_1')
                    conv_block3_conv5_1 = tf.layers.batch_normalization(conv_block3_conv5_1, training=self.phase, name='conv_block3_conv5_1_bn', reuse=self.reuse)
                    conv_block3_conv5_1 = tf.nn.relu(conv_block3_conv5_1)

                    conv_block3_conv5_2 = tf.layers.conv2d(conv_block3_conv5_1, filters=256, kernel_size=[3, 3], strides=[1, 1], padding='SAME', reuse=self.reuse, name='conv_block3_conv5_2')
                    conv_block3_conv5_2 = tf.layers.batch_normalization(conv_block3_conv5_2, training=self.phase, name='conv_block3_conv5_2_bn', reuse=self.reuse)
                    conv_block3_conv5_2 = tf.nn.relu(conv_block3_conv5_2)

                    conv_block3_conv5_3 = tf.layers.conv2d(conv_block3_conv5_2, filters=1024, kernel_size=[1, 1], strides=[1, 1], padding='SAME', reuse=self.reuse, name='conv_block3_conv5_3')
                    conv_block3_conv5_3 = tf.layers.batch_normalization(conv_block3_conv5_3, training=self.phase, name='conv_block3_conv5_3_bn', reuse=self.reuse)
                    # conv_block1_conv1_3 = tf.nn.relu(conv_block1_conv1_3)
                    conv_block3_output5 = conv_block3_output4 + conv_block3_conv5_3
                    conv_block3_output5 = tf.nn.relu(conv_block3_output5)

                with tf.variable_scope('block3_conv6') as scope:
                    # conv_block3_conv1_shortcut = tf.layers.conv2d(conv_block2_output4, filters=1024, kernel_size=[1,1], strides=[1,1], padding='SAME', reuse=self.reuse, name='conv_block3_conv1_shortcut')
                    # conv_block3_conv1_shortcut = tf.layers.batch_normalization(conv_block3_conv1_shortcut, training=self.phase, name='conv_block3_conv1_shortcut_bn', reuse=self.reuse)

                    conv_block3_conv6_1 = tf.layers.conv2d(conv_block3_output5, filters=256, kernel_size=[1, 1], strides=[1, 1], padding='SAME', reuse=self.reuse, name='conv_block3_conv6_1')
                    conv_block3_conv6_1 = tf.layers.batch_normalization(conv_block3_conv6_1, training=self.phase, name='conv_block3_conv6_1_bn', reuse=self.reuse)
                    conv_block3_conv6_1 = tf.nn.relu(conv_block3_conv6_1)

                    conv_block3_conv6_2 = tf.layers.conv2d(conv_block3_conv5_1, filters=256, kernel_size=[3, 3], strides=[1, 1], padding='SAME', reuse=self.reuse, name='conv_block3_conv6_2')
                    conv_block3_conv6_2 = tf.layers.batch_normalization(conv_block3_conv6_2, training=self.phase, name='conv_block3_conv6_2_bn', reuse=self.reuse)
                    conv_block3_conv6_2 = tf.nn.relu(conv_block3_conv6_2)

                    conv_block3_conv6_3 = tf.layers.conv2d(conv_block3_conv6_2, filters=1024, kernel_size=[1, 1], strides=[1, 1], padding='SAME', reuse=self.reuse, name='conv_block3_conv6_3')
                    conv_block3_conv6_3 = tf.layers.batch_normalization(conv_block3_conv6_3, training=self.phase, name='conv_block3_conv6_3_bn', reuse=self.reuse)
                    # conv_block1_conv1_3 = tf.nn.relu(conv_block1_conv1_3)
                    conv_block3_output6 = conv_block3_output5 + conv_block3_conv6_3
                    conv_block3_output6 = tf.nn.relu(conv_block3_output6)

            with tf.name_scope('block4'):
                with tf.variable_scope('block4_conv1') as scope:
                    conv_block4_conv1_shortcut = tf.layers.conv2d(conv_block3_output6, filters=2048, kernel_size=[1, 1], strides=[2, 2], padding='SAME', reuse=self.reuse, name='conv_block4_conv1_shortcut')
                    conv_block4_conv1_shortcut = tf.layers.batch_normalization(conv_block4_conv1_shortcut, training=self.phase, name='conv_block4_conv1_shortcut_bn', reuse=self.reuse)

                    conv_block4_conv1_1 = tf.layers.conv2d(conv_block3_output6, filters=512, kernel_size=[1, 1], strides=[2, 2], padding='SAME', reuse=self.reuse, name='conv_block4_conv1_1')
                    conv_block4_conv1_1 = tf.layers.batch_normalization(conv_block4_conv1_1, training=self.phase, name='conv_block4_conv1_1_bn', reuse=self.reuse)
                    conv_block4_conv1_1 = tf.nn.relu(conv_block4_conv1_1)

                    conv_block4_conv1_2 = tf.layers.conv2d(conv_block4_conv1_1, filters=512, kernel_size=[3, 3], strides=[1, 1], padding='SAME', reuse=self.reuse, name='conv_block4_conv1_2')
                    conv_block4_conv1_2 = tf.layers.batch_normalization(conv_block4_conv1_2, training=self.phase, name='conv_block4_conv1_2_bn', reuse=self.reuse)
                    conv_block4_conv1_2 = tf.nn.relu(conv_block4_conv1_2)

                    conv_block4_conv1_3 = tf.layers.conv2d(conv_block4_conv1_2, filters=2048, kernel_size=[1, 1], strides=[1, 1], padding='SAME', reuse=self.reuse, name='conv_block4_conv1_3')
                    conv_block4_conv1_3 = tf.layers.batch_normalization(conv_block4_conv1_3, training=self.phase, name='conv_block4_conv1_3_bn', reuse=self.reuse)
                    # conv_block1_conv1_3 = tf.nn.relu(conv_block1_conv1_3)
                    conv_block4_output1 = conv_block4_conv1_shortcut + conv_block4_conv1_3
                    conv_block4_output1 = tf.nn.relu(conv_block4_output1)

                with tf.variable_scope('block4_conv2') as scope:
                    # conv_block4_conv1_shortcut = tf.layers.conv2d(conv_block3_output6, filters=2048, kernel_size=[1,1], strides=[1,1], padding='SAME', reuse=self.reuse, name='conv_block4_conv1_shortcut')
                    # conv_block4_conv1_shortcut = tf.layers.batch_normalization(conv_block4_conv1_shortcut, training=self.phase, name='conv_block4_conv1_shortcut_bn', reuse=self.reuse)

                    conv_block4_conv2_1 = tf.layers.conv2d(conv_block4_output1, filters=512, kernel_size=[1, 1], strides=[1, 1], padding='SAME', reuse=self.reuse, name='conv_block4_conv2_1')
                    conv_block4_conv2_1 = tf.layers.batch_normalization(conv_block4_conv2_1, training=self.phase, name='conv_block4_conv2_1_bn', reuse=self.reuse)
                    conv_block4_conv2_1 = tf.nn.relu(conv_block4_conv2_1)

                    conv_block4_conv2_2 = tf.layers.conv2d(conv_block4_conv2_1, filters=512, kernel_size=[3, 3], strides=[1, 1], padding='SAME', reuse=self.reuse, name='conv_block4_conv2_2')
                    conv_block4_conv2_2 = tf.layers.batch_normalization(conv_block4_conv2_2, training=self.phase, name='conv_block4_conv2_2_bn', reuse=self.reuse)
                    conv_block4_conv2_2 = tf.nn.relu(conv_block4_conv2_2)

                    conv_block4_conv2_3 = tf.layers.conv2d(conv_block4_conv2_2, filters=2048, kernel_size=[1, 1], strides=[1, 1], padding='SAME', reuse=self.reuse, name='conv_block4_conv2_3')
                    conv_block4_conv2_3 = tf.layers.batch_normalization(conv_block4_conv2_3, training=self.phase, name='conv_block4_conv2_3_bn', reuse=self.reuse)
                    # conv_block1_conv1_3 = tf.nn.relu(conv_block1_conv1_3)
                    conv_block4_output2 = conv_block4_output1 + conv_block4_conv2_3
                    conv_block4_output2 = tf.nn.relu(conv_block4_output2)

                with tf.variable_scope('block4_conv3') as scope:
                    # conv_block4_conv1_shortcut = tf.layers.conv2d(conv_block3_output6, filters=2048, kernel_size=[1,1], strides=[1,1], padding='SAME', reuse=self.reuse, name='conv_block4_conv1_shortcut')
                    # conv_block4_conv1_shortcut = tf.layers.batch_normalization(conv_block4_conv1_shortcut, training=self.phase, name='conv_block4_conv1_shortcut_bn', reuse=self.reuse)

                    conv_block4_conv3_1 = tf.layers.conv2d(conv_block4_output2, filters=512, kernel_size=[1, 1], strides=[1, 1], padding='SAME', reuse=self.reuse, name='conv_block4_conv3_1')
                    conv_block4_conv3_1 = tf.layers.batch_normalization(conv_block4_conv3_1, training=self.phase, name='conv_block4_conv3_1_bn', reuse=self.reuse)
                    conv_block4_conv3_1 = tf.nn.relu(conv_block4_conv3_1)

                    conv_block4_conv3_2 = tf.layers.conv2d(conv_block4_conv3_1, filters=512, kernel_size=[3, 3], strides=[1, 1], padding='SAME', reuse=self.reuse, name='conv_block4_conv3_2')
                    conv_block4_conv3_2 = tf.layers.batch_normalization(conv_block4_conv3_2, training=self.phase, name='conv_block4_conv3_2_bn', reuse=self.reuse)
                    conv_block4_conv3_2 = tf.nn.relu(conv_block4_conv3_2)

                    conv_block4_conv3_3 = tf.layers.conv2d(conv_block4_conv3_2, filters=2048, kernel_size=[1, 1], strides=[1, 1], padding='SAME', reuse=self.reuse, name='conv_block4_conv3_3')
                    conv_block4_conv3_3 = tf.layers.batch_normalization(conv_block4_conv3_3, training=self.phase, name='conv_block4_conv3_3_bn', reuse=self.reuse)
                    # conv_block1_conv1_3 = tf.nn.relu(conv_block1_conv1_3)
                    conv_block4_output3 = conv_block4_output2 + conv_block4_conv3_3
                    conv_block4_output3 = tf.nn.relu(conv_block4_output3)

            with tf.variable_scope('fc'):
                fc1 = tf.layers.conv2d(conv_block4_output3, filters=2048, kernel_size=[16, 1], strides=[1, 1], padding='VALID', reuse=self.reuse, name='fc1')
                fc2 = tf.reduce_mean(fc1, axis=[1, 2], name='avgpool')
                flattened = tf.contrib.layers.flatten(fc2)
                flattened = tf.nn.l2_normalize(flattened)
                w = tf.Variable(tf.truncated_normal([2048, 1024], stddev=0.1), name="w")
                b = tf.Variable(tf.constant(0.1, shape=[1024]), name="b")
                h = tf.nn.xw_plus_b(flattened, w, b, name="scores")
                h = tf.nn.relu(h, name="relu")

                with tf.name_scope("dropout"):
                    h = tf.nn.dropout(h, self.dropout_keep_prob)

            self.embedding = self.graph.get_tensor_by_name('fc/scores:0')[0]

            with tf.variable_scope('output'):
                w = tf.get_variable('w', shape=[1024, n_classes], initializer=tf.contrib.layers.xavier_initializer())
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

