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

class ResNet50Vox(Model):

    """
       Class to represent Speaker Verification (SV) model based on the ResNet50 architecture - Embedding vectors of size 512 are returned
       Chung, J. S., Nagrani, A., & Zisserman, A. (2018).
       VoxCeleb2: Deep Speaker Recognition.
       Proc. Interspeech 2018, 1086-1090.
    """

    def __init__(self, name='resnet50vox', id='', noises=None, cache=None, n_seconds=3, sample_rate=16000):
        super().__init__(name, id, noises, cache, n_seconds, sample_rate)

    def __conv_bn_dynamic_apool(self, inp_tensor, layer_idx, conv_filters, conv_kernel_size, conv_strides, conv_pad, conv_layer_prefix='conv'):
        x = tf.keras.layers.ZeroPadding2D(padding=conv_pad, name='pad{}'.format(layer_idx))(inp_tensor)
        x = tf.keras.layers.Conv2D(filters=conv_filters, kernel_size=conv_kernel_size, strides=conv_strides, padding='valid', name='{}{}'.format(conv_layer_prefix, layer_idx))(x)
        x = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=1., name='bn{}'.format(layer_idx))(x)
        x = tf.keras.layers.Activation('relu', name='relu{}'.format(layer_idx))(x)
        x = tf.keras.layers.AveragePooling2D(pool_size=(1, 8), strides=(1,1), name='gapool{}'.format(layer_idx))(x)
        x = tf.keras.layers.Reshape((1, 1, conv_filters), name='reshape{}'.format(layer_idx))(x)
        return x

    def build(self, classes=None):
        super().build(classes)
        print('>', 'building', self.name, 'model on', classes, 'classes')

        signal_input = tf.keras.Input(shape=(None,1,))
        impulse_input = tf.keras.Input(shape=(3,))

        x = tf.keras.layers.Lambda(lambda x: play_n_rec(x, self.noises, self.cache), name='play_n_rec')([signal_input, impulse_input])
        x = tf.keras.layers.Lambda(lambda x: get_tf_spectrum(x, self.sample_rate), name='acoustic_layer')(x)

        # Conv1
        conv0_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=[7, 7], strides=[2, 2], padding='SAME', name='conv1')(x)
        conv0_1 = tf.keras.layers.BatchNormalization(name='bbn0')(conv0_1)
        conv0_1 = tf.keras.layers.ReLU()(conv0_1)

        # Pool 1
        conv0_1 = tf.keras.layers.MaxPool2D(pool_size=[3, 3], strides=[2, 2], name='mpool1')(conv0_1)

        # Conv 2_x
        conv_block1_conv1_shortcut = tf.keras.layers.Conv2D(filters=256, kernel_size=[1, 1], strides=[1, 1], padding='VALID', name='conv_block1_conv1_shortcut_conv')(conv0_1)
        conv_block1_conv1_shortcut = tf.keras.layers.BatchNormalization(name='conv_block1_conv1_shortcut_bn')(conv_block1_conv1_shortcut)

        conv_block1_conv1_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=[1, 1], strides=[1, 1], padding='VALID', name='conv_block1_conv1_1')(conv0_1)
        conv_block1_conv1_1 = tf.keras.layers.BatchNormalization(name='conv_block1_conv1_1_bn')(conv_block1_conv1_1)
        conv_block1_conv1_1 = tf.keras.layers.ReLU()(conv_block1_conv1_1)

        conv_block1_conv1_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding='SAME', name='conv_block1_conv1_2')(conv_block1_conv1_1)
        conv_block1_conv1_2 = tf.keras.layers.BatchNormalization(name='conv_block1_conv1_2_bn')(conv_block1_conv1_2)
        conv_block1_conv1_2 = tf.keras.layers.ReLU()(conv_block1_conv1_2)

        conv_block1_conv1_3 = tf.keras.layers.Conv2D(filters=256, kernel_size=[1, 1], strides=[1, 1], padding='VALID', name='conv_block1_conv1_3')(conv_block1_conv1_2)
        conv_block1_conv1_3 = tf.keras.layers.BatchNormalization(name='conv_block1_conv1_3_bn')(conv_block1_conv1_3)
        conv_block1_output1 = tf.keras.layers.Add()([conv_block1_conv1_shortcut, conv_block1_conv1_3])
        conv_block1_output1 = tf.keras.layers.ReLU()(conv_block1_output1)

        conv_block1_conv2_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=[1, 1], strides=[1, 1], padding='SAME', name='conv_block1_conv2_1')(conv_block1_output1)
        conv_block1_conv2_1 = tf.keras.layers.BatchNormalization(name='conv_block1_conv2_1_bn')(conv_block1_conv2_1)
        conv_block1_conv2_1 = tf.keras.layers.ReLU()(conv_block1_conv2_1)

        conv_block1_conv2_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding='SAME', name='conv_block1_conv2_2')(conv_block1_conv2_1)
        conv_block1_conv2_2 = tf.keras.layers.BatchNormalization(name='conv_block1_conv2_2_bn')(conv_block1_conv2_2)
        conv_block1_conv2_2 = tf.keras.layers.ReLU()(conv_block1_conv2_2)

        conv_block1_conv2_3 = tf.keras.layers.Conv2D(filters=256, kernel_size=[1, 1], strides=[1, 1], padding='SAME', name='conv_block1_conv2_3')(conv_block1_conv2_2)
        conv_block1_conv2_3 = tf.keras.layers.BatchNormalization(name='conv_block1_conv2_3_bn')(conv_block1_conv2_3)
        conv_block1_output2 = tf.keras.layers.Add()([conv_block1_output1, conv_block1_conv2_3])
        conv_block1_output2 = tf.keras.layers.ReLU()(conv_block1_output2)

        conv_block1_conv3_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=[1, 1], strides=[1, 1], padding='SAME', name='conv_block1_conv3_1')(conv_block1_output2)
        conv_block1_conv3_1 = tf.keras.layers.BatchNormalization(name='conv_block1_conv3_1_bn')(conv_block1_conv3_1)
        conv_block1_conv3_1 = tf.keras.layers.ReLU()(conv_block1_conv3_1)

        conv_block1_conv3_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding='SAME', name='conv_block1_conv3_2')(conv_block1_conv3_1)
        conv_block1_conv3_2 = tf.keras.layers.BatchNormalization(name='conv_block1_conv3_2_bn')(conv_block1_conv3_2)
        conv_block1_conv3_2 = tf.keras.layers.ReLU()(conv_block1_conv3_2)

        conv_block1_conv3_3 = tf.keras.layers.Conv2D(filters=256, kernel_size=[1, 1], strides=[1, 1], padding='SAME', name='conv_block1_conv3_3')(conv_block1_conv3_2)
        conv_block1_conv3_3 = tf.keras.layers.BatchNormalization(name='cconv_block1_conv3_3_bn')(conv_block1_conv3_3)
        conv_block1_output3 = tf.keras.layers.Add()([conv_block1_output2, conv_block1_conv3_3])
        conv_block1_output3 = tf.keras.layers.ReLU()(conv_block1_output3)

        # Conv 3_x
        conv_block2_conv1_shortcut = tf.keras.layers.Conv2D(filters=512, kernel_size=[1, 1], strides=[2, 2], padding='SAME', name='conv_block2_conv1_shortcut_conv')(conv_block1_output3)
        conv_block2_conv1_shortcut = tf.keras.layers.BatchNormalization(name='conv_block2_conv1_shortcut_bn')(conv_block2_conv1_shortcut)

        conv_block2_conv1_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=[1, 1], strides=[2, 2], padding='SAME', name='conv_block2_conv1_1')(conv_block1_output3)
        conv_block2_conv1_1 = tf.keras.layers.BatchNormalization(name='conv_block2_conv1_1_bn')(conv_block2_conv1_1)
        conv_block2_conv1_1 = tf.keras.layers.ReLU()(conv_block2_conv1_1)

        conv_block2_conv1_2 = tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], strides=[1, 1], padding='SAME', name='conv_block2_conv1_2')(conv_block2_conv1_1)
        conv_block2_conv1_2 = tf.keras.layers.BatchNormalization(name='conv_block2_conv1_2_bn')(conv_block2_conv1_2)
        conv_block2_conv1_2 = tf.keras.layers.ReLU()(conv_block2_conv1_2)

        conv_block2_conv1_3 = tf.keras.layers.Conv2D(filters=512, kernel_size=[1, 1], strides=[1, 1], padding='SAME', name='conv_block2_conv1_3')(conv_block2_conv1_2)
        conv_block2_conv1_3 = tf.keras.layers.BatchNormalization(name='conv_block2_conv1_3_bn')(conv_block2_conv1_3)
        conv_block2_output1 = tf.keras.layers.Add()([conv_block2_conv1_shortcut, conv_block2_conv1_3])
        conv_block2_output1 = tf.keras.layers.ReLU()(conv_block2_output1)

        conv_block2_conv2_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=[1, 1], strides=[1, 1], padding='SAME', name='conv_block2_conv2_1')(conv_block2_output1)
        conv_block2_conv2_1 = tf.keras.layers.BatchNormalization(name='conv_block2_conv2_1_bn')(conv_block2_conv2_1)
        conv_block2_conv2_1 = tf.keras.layers.ReLU()(conv_block2_conv2_1)

        conv_block2_conv2_2 = tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], strides=[1, 1], padding='SAME', name='conv_block2_conv2_2')(conv_block2_conv2_1)
        conv_block2_conv2_2 = tf.keras.layers.BatchNormalization(name='conv_block2_conv2_2_bn')(conv_block2_conv2_2)
        conv_block2_conv2_2 = tf.keras.layers.ReLU()(conv_block2_conv2_2)

        conv_block2_conv2_3 = tf.keras.layers.Conv2D(filters=512, kernel_size=[1, 1], strides=[1, 1], padding='SAME', name='conv_block2_conv2_3')(conv_block2_conv2_2)
        conv_block2_conv2_3 = tf.keras.layers.BatchNormalization(name='conv_block2_conv2_3_bn')(conv_block2_conv2_3)
        conv_block2_output2 = tf.keras.layers.Add()([conv_block2_output1, conv_block2_conv2_3])
        conv_block2_output2 = tf.keras.layers.ReLU()(conv_block2_output2)

        conv_block2_conv3_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=[1, 1], strides=[1, 1], padding='SAME', name='conv_block2_conv3_1')(conv_block2_output2)
        conv_block2_conv3_1 = tf.keras.layers.BatchNormalization(name='conv_block2_conv3_1_bn')(conv_block2_conv3_1)
        conv_block2_conv3_1 = tf.keras.layers.ReLU()(conv_block2_conv3_1)

        conv_block2_conv3_2 = tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], strides=[1, 1], padding='SAME', name='conv_block2_conv3_2')(conv_block2_conv3_1)
        conv_block2_conv3_2 = tf.keras.layers.BatchNormalization(name='conv_block2_conv3_2_bn')(conv_block2_conv3_2)
        conv_block2_conv3_2 = tf.keras.layers.ReLU()(conv_block2_conv3_2)

        conv_block2_conv3_3 = tf.keras.layers.Conv2D(filters=512, kernel_size=[1, 1], strides=[1, 1], padding='SAME', name='conv_block2_conv3_3')(conv_block2_conv3_2)
        conv_block2_conv3_3 = tf.keras.layers.BatchNormalization(name='conv_block2_conv3_3_bn')(conv_block2_conv3_3)
        conv_block2_output3 = tf.keras.layers.Add()([conv_block2_output2, conv_block2_conv3_3])
        conv_block2_output3 = tf.keras.layers.ReLU()(conv_block2_output3)

        conv_block2_conv4_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=[1, 1], strides=[1, 1], padding='SAME', name='conv_block2_conv4_1')(conv_block2_output3)
        conv_block2_conv4_1 = tf.keras.layers.BatchNormalization(name='conv_block2_conv4_1_bn')(conv_block2_conv4_1)
        conv_block2_conv4_1 = tf.keras.layers.ReLU()(conv_block2_conv4_1)

        conv_block2_conv4_2 = tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], strides=[1, 1], padding='SAME', name='conv_block2_conv4_2')(conv_block2_conv4_1)
        conv_block2_conv4_2 = tf.keras.layers.BatchNormalization(name='conv_block2_conv4_2_bn')(conv_block2_conv4_2)
        conv_block2_conv4_2 = tf.keras.layers.ReLU()(conv_block2_conv4_2)

        conv_block2_conv4_3 = tf.keras.layers.Conv2D(filters=512, kernel_size=[1, 1], strides=[1, 1], padding='SAME', name='conv_block2_conv4_3')(conv_block2_conv4_2)
        conv_block2_conv4_3 = tf.keras.layers.BatchNormalization(name='conv_block2_conv4_3_bn')(conv_block2_conv4_3)
        conv_block2_output4 = tf.keras.layers.Add()([conv_block2_output3, conv_block2_conv4_3])
        conv_block2_output4 = tf.keras.layers.ReLU()(conv_block2_output4)

        # Conv 4_x
        conv_block3_conv1_shortcut = tf.keras.layers.Conv2D(filters=1024, kernel_size=[1, 1], strides=[2, 2], padding='SAME', name='conv_block3_conv1_shortcut')(conv_block2_output4)
        conv_block3_conv1_shortcut = tf.keras.layers.BatchNormalization(name='conv_block3_conv1_shortcut_bn')(conv_block3_conv1_shortcut)

        conv_block3_conv1_1 = tf.keras.layers.Conv2D(filters=256, kernel_size=[1, 1], strides=[2, 2], padding='SAME', name='conv_block3_conv1_1')(conv_block2_output4)
        conv_block3_conv1_1 = tf.keras.layers.BatchNormalization(name='conv_block3_conv1_1_bn')(conv_block3_conv1_1)
        conv_block3_conv1_1 = tf.keras.layers.ReLU()(conv_block3_conv1_1)

        conv_block3_conv1_2 = tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding='SAME', name='conv_block3_conv1_2')(conv_block3_conv1_1)
        conv_block3_conv1_2 = tf.keras.layers.BatchNormalization(name='conv_block3_conv1_2_bn')(conv_block3_conv1_2)
        conv_block3_conv1_2 = tf.keras.layers.ReLU()(conv_block3_conv1_2)

        conv_block3_conv1_3 = tf.keras.layers.Conv2D(filters=1024, kernel_size=[1, 1], strides=[1, 1], padding='SAME', name='conv_block3_conv1_3')(conv_block3_conv1_2)
        conv_block3_conv1_3 = tf.keras.layers.BatchNormalization(name='conv_block3_conv1_3_bn')(conv_block3_conv1_3)
        conv_block3_output1 = tf.keras.layers.Add()([conv_block3_conv1_shortcut, conv_block3_conv1_3])
        conv_block3_output1 = tf.keras.layers.ReLU()(conv_block3_output1)

        conv_block3_conv2_1 = tf.keras.layers.Conv2D(filters=256, kernel_size=[1, 1], strides=[1, 1], padding='SAME', name='conv_block3_conv2_1')(conv_block3_output1)
        conv_block3_conv2_1 = tf.keras.layers.BatchNormalization(name='conv_block3_conv2_1_bn')(conv_block3_conv2_1)
        conv_block3_conv2_1 = tf.keras.layers.ReLU()(conv_block3_conv2_1)

        conv_block3_conv2_2 = tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding='SAME', name='conv_block3_conv2_2')(conv_block3_conv2_1)
        conv_block3_conv2_2 = tf.keras.layers.BatchNormalization(name='conv_block3_conv2_2_bn')(conv_block3_conv2_2)
        conv_block3_conv2_2 = tf.keras.layers.ReLU()(conv_block3_conv2_2)

        conv_block3_conv2_3 = tf.keras.layers.Conv2D(filters=1024, kernel_size=[1, 1], strides=[1, 1], padding='SAME', name='conv_block3_conv2_3')(conv_block3_conv2_2)
        conv_block3_conv2_3 = tf.keras.layers.BatchNormalization(name='conv_block3_conv2_3_bn')(conv_block3_conv2_3)
        conv_block3_output2 = tf.keras.layers.Add()([conv_block3_output1, conv_block3_conv2_3])
        conv_block3_output2 = tf.keras.layers.ReLU()(conv_block3_output2)

        conv_block3_conv3_1 = tf.keras.layers.Conv2D(filters=256, kernel_size=[1, 1], strides=[1, 1], padding='SAME', name='conv_block3_conv3_1')(conv_block3_output2)
        conv_block3_conv3_1 = tf.keras.layers.BatchNormalization(name='conv_block3_conv3_1_1_bn')(conv_block3_conv3_1)
        conv_block3_conv3_1 = tf.keras.layers.ReLU()(conv_block3_conv3_1)

        conv_block3_conv3_2 = tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding='SAME', name='conv_block3_conv3_2')(conv_block3_conv3_1)
        conv_block3_conv3_2 = tf.keras.layers.BatchNormalization(name='conv_block3_conv3_2_bn')(conv_block3_conv3_2)
        conv_block3_conv3_2 = tf.keras.layers.ReLU()(conv_block3_conv3_2)

        conv_block3_conv3_3 = tf.keras.layers.Conv2D(filters=1024, kernel_size=[1, 1], strides=[1, 1], padding='SAME', name='conv_block3_conv3_3')(conv_block3_conv3_2)
        conv_block3_conv3_3 = tf.keras.layers.BatchNormalization(name='conv_block3_conv3_3_bn')(conv_block3_conv3_3)
        conv_block3_output3 = tf.keras.layers.Add()([conv_block3_output2, conv_block3_conv3_3])
        conv_block3_output3 = tf.keras.layers.ReLU()(conv_block3_output3)

        conv_block3_conv4_1 = tf.keras.layers.Conv2D(filters=256, kernel_size=[1, 1], strides=[1, 1], padding='SAME', name='conv_block3_conv4_1')(conv_block3_output3)
        conv_block3_conv4_1 = tf.keras.layers.BatchNormalization(name='conv_block3_conv4_1_bn')(conv_block3_conv4_1)
        conv_block3_conv4_1 = tf.keras.layers.ReLU()(conv_block3_conv4_1)

        conv_block3_conv4_2 = tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding='SAME', name='conv_block3_conv4_2')(conv_block3_conv4_1)
        conv_block3_conv4_2 = tf.keras.layers.BatchNormalization(name='conv_block3_conv4_2_bn')(conv_block3_conv4_2)
        conv_block3_conv4_2 = tf.keras.layers.ReLU()(conv_block3_conv4_2)

        conv_block3_conv4_3 = tf.keras.layers.Conv2D(filters=1024, kernel_size=[1, 1], strides=[1, 1], padding='SAME', name='conv_block3_conv4_3')(conv_block3_conv4_2)
        conv_block3_conv4_3 = tf.keras.layers.BatchNormalization(name='conv_block3_conv4_3_bn')(conv_block3_conv4_3)
        conv_block3_output4 = tf.keras.layers.Add()([conv_block3_output3, conv_block3_conv4_3])
        conv_block3_output4 = tf.keras.layers.ReLU()(conv_block3_output4)

        conv_block3_conv5_1 = tf.keras.layers.Conv2D(filters=256, kernel_size=[1, 1], strides=[1, 1], padding='SAME', name='conv_block3_conv5_1')(conv_block3_output4)
        conv_block3_conv5_1 = tf.keras.layers.BatchNormalization(name='conv_block3_conv5_1_bn')(conv_block3_conv5_1)
        conv_block3_conv5_1 = tf.keras.layers.ReLU()(conv_block3_conv5_1)

        conv_block3_conv5_2 = tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding='SAME', name='conv_block3_conv5_2')(conv_block3_conv5_1)
        conv_block3_conv5_2 = tf.keras.layers.BatchNormalization(name='conv_block3_conv5_2_bn')(conv_block3_conv5_2)
        conv_block3_conv5_2 = tf.keras.layers.ReLU()(conv_block3_conv5_2)

        conv_block3_conv5_3 = tf.keras.layers.Conv2D(filters=1024, kernel_size=[1, 1], strides=[1, 1], padding='SAME', name='conv_block3_conv5_3')(conv_block3_conv5_2)
        conv_block3_conv5_3 = tf.keras.layers.BatchNormalization(name='conv_block3_conv5_3_bn')(conv_block3_conv5_3)
        conv_block3_output5 = tf.keras.layers.Add()([conv_block3_output4, conv_block3_conv5_3])
        conv_block3_output5 = tf.keras.layers.ReLU()(conv_block3_output5)

        conv_block3_conv6_1 = tf.keras.layers.Conv2D(filters=256, kernel_size=[1, 1], strides=[1, 1], padding='SAME', name='conv_block3_conv6_1')(conv_block3_output5)
        conv_block3_conv6_1 = tf.keras.layers.BatchNormalization(name='conv_block3_conv6_1_bn')(conv_block3_conv6_1)
        conv_block3_conv6_1 = tf.keras.layers.ReLU()(conv_block3_conv6_1)

        conv_block3_conv6_2 = tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1], padding='SAME', name='conv_block3_conv6_2')(conv_block3_conv6_1)
        conv_block3_conv6_2 = tf.keras.layers.BatchNormalization(name='conv_block3_conv6_2_bn')(conv_block3_conv6_2)
        conv_block3_conv6_2 = tf.keras.layers.ReLU()(conv_block3_conv6_2)

        conv_block3_conv6_3 = tf.keras.layers.Conv2D(filters=1024, kernel_size=[1, 1], strides=[1, 1], padding='SAME', name='conv_block3_conv6_3')(conv_block3_conv6_2)
        conv_block3_conv6_3 = tf.keras.layers.BatchNormalization(name='conv_block3_conv6_3_bn')(conv_block3_conv6_3)
        conv_block3_output6 = tf.keras.layers.Add()([conv_block3_output5, conv_block3_conv6_3])
        conv_block3_output6 = tf.keras.layers.ReLU()(conv_block3_output6)

        # Conv 5_x
        conv_block4_conv1_shortcut = tf.keras.layers.Conv2D(filters=2048, kernel_size=[1, 1], strides=[2, 2], padding='SAME', name='conv_block4_conv1_shortcut')(conv_block3_output6)
        conv_block4_conv1_shortcut = tf.keras.layers.BatchNormalization(name='conv_block4_conv1_shortcut_bn')(conv_block4_conv1_shortcut)

        conv_block4_conv1_1 = tf.keras.layers.Conv2D(filters=512, kernel_size=[1, 1], strides=[2, 2], padding='SAME', name='conv_block4_conv1_1')(conv_block3_output6)
        conv_block4_conv1_1 = tf.keras.layers.BatchNormalization(name='conv_block4_conv1_1_bn')(conv_block4_conv1_1)
        conv_block4_conv1_1 = tf.keras.layers.ReLU()(conv_block4_conv1_1)

        conv_block4_conv1_2 = tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], strides=[1, 1], padding='SAME', name='conv_block4_conv1_2')(conv_block4_conv1_1)
        conv_block4_conv1_2 = tf.keras.layers.BatchNormalization(name='conv_block4_conv1_2_bn')(conv_block4_conv1_2)
        conv_block4_conv1_2 = tf.keras.layers.ReLU()(conv_block4_conv1_2)

        conv_block4_conv1_3 = tf.keras.layers.Conv2D(filters=2048, kernel_size=[1, 1], strides=[1, 1], padding='SAME', name='conv_block4_conv1_3')(conv_block4_conv1_2)
        conv_block4_conv1_3 = tf.keras.layers.BatchNormalization(name='conv_block4_conv1_3_bn')(conv_block4_conv1_3)
        conv_block4_output1 = tf.keras.layers.Add()([conv_block4_conv1_shortcut, conv_block4_conv1_3])
        conv_block4_output1 = tf.keras.layers.ReLU()(conv_block4_output1)

        conv_block4_conv2_1 = tf.keras.layers.Conv2D(filters=512, kernel_size=[1, 1], strides=[1, 1], padding='SAME', name='conv_block4_conv2_1')(conv_block4_output1)
        conv_block4_conv2_1 = tf.keras.layers.BatchNormalization(name='conv_block4_conv2_1_bn')(conv_block4_conv2_1)
        conv_block4_conv2_1 = tf.keras.layers.ReLU()(conv_block4_conv2_1)

        conv_block4_conv2_2 = tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], strides=[1, 1], padding='SAME', name='conv_block4_conv2_2')(conv_block4_conv2_1)
        conv_block4_conv2_2 = tf.keras.layers.BatchNormalization(name='conv_block4_conv2_2_bn')(conv_block4_conv2_2)
        conv_block4_conv2_2 = tf.keras.layers.ReLU()(conv_block4_conv2_2)

        conv_block4_conv2_3 = tf.keras.layers.Conv2D(filters=2048, kernel_size=[1, 1], strides=[1, 1], padding='SAME', name='conv_block4_conv2_3')(conv_block4_conv2_2)
        conv_block4_conv2_3 = tf.keras.layers.BatchNormalization(name='conv_block4_conv2_3_bn')(conv_block4_conv2_3)
        conv_block4_output2 = tf.keras.layers.Add()([conv_block4_output1, conv_block4_conv2_3])
        conv_block4_output2 = tf.keras.layers.ReLU()(conv_block4_output2)

        conv_block4_conv3_1 = tf.keras.layers.Conv2D(filters=512, kernel_size=[1, 1], strides=[1, 1], padding='SAME', name='conv_block4_conv3_1')(conv_block4_output2)
        conv_block4_conv3_1 = tf.keras.layers.BatchNormalization(name='conv_block4_conv3_1_bn')(conv_block4_conv3_1)
        conv_block4_conv3_1 = tf.keras.layers.ReLU()(conv_block4_conv3_1)

        conv_block4_conv3_2 = tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], strides=[1, 1], padding='SAME', name='conv_block4_conv3_2')(conv_block4_conv3_1)
        conv_block4_conv3_2 = tf.keras.layers.BatchNormalization(name='conv_block4_conv3_2_bn')(conv_block4_conv3_2)
        conv_block4_conv3_2 = tf.keras.layers.ReLU()(conv_block4_conv3_2)

        conv_block4_conv3_3 = tf.keras.layers.Conv2D(filters=2048, kernel_size=[1, 1], strides=[1, 1], padding='SAME', name='conv_block4_conv3_3')(conv_block4_conv3_2)
        conv_block4_conv3_3 = tf.keras.layers.BatchNormalization(name='conv_block4_conv3_3_bn')(conv_block4_conv3_3)
        conv_block4_output3 = tf.keras.layers.Add()([conv_block4_output2, conv_block4_conv3_3])
        conv_block4_output3 = tf.keras.layers.ReLU()(conv_block4_output3)

        # Fc layers
        embedding_layer = self.__conv_bn_dynamic_apool(x, layer_idx=6, conv_filters=2048, conv_kernel_size=(9, 1), conv_strides=(1, 1), conv_pad=(0, 0), conv_layer_prefix='fc')
        x = tf.keras.layers.Lambda(lambda y: tf.keras.backend.l2_normalize(y, axis=3), name='norm')(embedding_layer)
        output = tf.keras.layers.Conv2D(filters=classes, kernel_size=(1, 1), strides=(1, 1), padding='valid', name='fc8')(x)

        self.model = tf.keras.Model(inputs=[signal_input, impulse_input], outputs=[output])
        print('>', 'built', self.name, 'model on', classes, 'classes')

        super().load()

        self.inference_model = tf.keras.Model(inputs=[signal_input, impulse_input], outputs=[embedding_layer])
        print('>', 'built', self.name, 'inference model')

        self.model.summary()