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

class ResNet34Vox(Model):

    def __init__(self, name='resnet34vox', id='', noises=None, cache=None, n_seconds=3, sample_rate=16000):
        super().__init__(name, id, noises, cache, n_seconds, sample_rate)

    def __conv_bn_dynamic_apool(self, inp_tensor, layer_idx, conv_filters, conv_kernel_size, conv_strides, conv_pad, conv_layer_prefix='conv'):
        x = tf.keras.layers.ZeroPadding2D(padding=conv_pad, name='pad{}'.format(layer_idx))(inp_tensor)
        x = tf.keras.layers.Conv2D(filters=conv_filters, kernel_size=conv_kernel_size, strides=conv_strides, padding='valid', name='{}{}'.format(conv_layer_prefix, layer_idx))(x)
        x = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=1., name='bn{}'.format(layer_idx))(x)
        x = tf.keras.layers.Activation('relu', name='relu{}'.format(layer_idx))(x)
        x = tf.keras.layers.AveragePooling2D(pool_size=(1, 8), strides=(1,1), name='gapool{}'.format(layer_idx))(x)
        x = tf.keras.layers.Reshape((1, 1, conv_filters), name='reshape{}'.format(layer_idx))(x)
        return x

    def identity_block2d(self, input_tensor, kernel_size, filters, stage, block, kernel_initializer):
        filters1, filters2, filters3 = filters

        conv_name_1 = 'conv' + str(stage) + '_' + str(block) + '_1x1_reduce'
        bn_name_1 = 'bn' + str(stage) + '_' + str(block) + '_1x1_reduce'

        x = tf.keras.layers.Conv2D(filters=filters1, kernel_size=(1, 1), use_bias=False, kernel_initializer=kernel_initializer, name=conv_name_1)(input_tensor)
        x = tf.keras.layers.BatchNormalization(name=bn_name_1)(x)
        x = tf.keras.layers.ReLU()(x)

        conv_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3'
        bn_name_2 = 'bn' + str(stage) + '_' + str(block) + '_3x3'

        x = tf.keras.layers.Conv2D(filters=filters2, kernel_size=kernel_size, use_bias=False, padding='SAME', kernel_initializer=kernel_initializer, name=conv_name_2)(x)
        x = tf.keras.layers.BatchNormalization(name=bn_name_2)(x)
        x = tf.keras.layers.ReLU()(x)

        conv_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase'
        bn_name_3 = 'bn' + str(stage) + '_' + str(block) + '_1x1_increase'

        x = tf.keras.layers.Conv2D(filters=filters3, kernel_size=(1, 1), use_bias=False, kernel_initializer=kernel_initializer, name=conv_name_3)(x)
        x = tf.keras.layers.BatchNormalization(name=bn_name_3)(x)

        x = tf.keras.layers.Add()([input_tensor, x])
        x = tf.keras.layers.ReLU()(x)

        return x

    def conv_block_2d(self, input_tensor, kernel_size, filters, stage, block, strides, kernel_initializer):
        filters1, filters2, filters3 = filters

        conv_name_1 = 'conv' + str(stage) + '_' + str(block) + '_1x1_reduce'
        bn_name_1 = 'bn' + str(stage) + '_' + str(block) + '_1x1_reduce'

        x = tf.keras.layers.Conv2D(filters=filters1, kernel_size=(1, 1), strides=strides, use_bias=False, kernel_initializer=kernel_initializer, name=conv_name_1)(input_tensor)
        x = tf.keras.layers.BatchNormalization(name=bn_name_1)(x)
        x = tf.keras.layers.ReLU()(x)

        conv_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3'
        bn_name_2 = 'bn' + str(stage) + '_' + str(block) + '_3x3'

        x = tf.keras.layers.Conv2D(filters=filters2, kernel_size=kernel_size, padding='SAME', use_bias=False, kernel_initializer=kernel_initializer, name=conv_name_2)(x)
        x = tf.keras.layers.BatchNormalization(name=bn_name_2)(x)
        x = tf.keras.layers.ReLU()(x)

        conv_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase'
        bn_name_3 = 'bn' + str(stage) + '_' + str(block) + '_1x1_increase'

        x = tf.keras.layers.Conv2D(filters=filters3, kernel_size=(1, 1), use_bias=False, kernel_initializer=kernel_initializer, name=conv_name_3)(x)
        x = tf.keras.layers.BatchNormalization(name=bn_name_3)(x)

        conv_name_4 = 'conv' + str(stage) + '_' + str(block) + '_1x1_shortcut'
        bn_name_4 = 'bn' + str(stage) + '_' + str(block) + '_1x1_shortcut'

        shortcut = tf.keras.layers.Conv2D(filters=filters3, kernel_size=(1, 1), use_bias=False, strides=strides, kernel_initializer=kernel_initializer, name=conv_name_4)(input_tensor)
        shortcut = tf.keras.layers.BatchNormalization(name=bn_name_4)(shortcut)

        x = tf.keras.layers.Add()([shortcut, x])
        x = tf.keras.layers.ReLU()(x)

        return x

    def build(self, classes=None):
        super().build(classes)
        print('>', 'building', self.name, 'model on', classes, 'classes')

        kernel_initializer = tf.initializers.GlorotUniform()

        signal_input = tf.keras.Input(shape=(None,1,))
        impulse_input = tf.keras.Input(shape=(3,))

        x = tf.keras.layers.Lambda(lambda x: play_n_rec(x, self.noises, self.cache), name='play_n_rec')([signal_input, impulse_input])
        x = tf.keras.layers.Lambda(lambda x: get_tf_spectrum(x, self.sample_rate), name='acoustic_layer')(x)

        # Conv 1
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), kernel_initializer=kernel_initializer, use_bias=False, padding='SAME', name='voice_conv1_1/3x3_s1')(x)
        x = tf.keras.layers.BatchNormalization(name='voice_bn1_1/3x3_s1')(x)
        x = tf.keras.layers.ReLU()(x)

        # Pool 1
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='voice_mpool1')(x)

        # Conv 2_x
        x1 = self.conv_block_2d(x, kernel_size=3, filters=[48, 48, 96], stage=2, block='voice_1a', strides=(1, 1), kernel_initializer=kernel_initializer)
        x1 = self.identity_block2d(x1, kernel_size=3, filters=[48, 48, 96], stage=2, block='voice_1b', kernel_initializer=kernel_initializer)

        # Conv 3_x
        x2 = self.conv_block_2d(x1, kernel_size=3, filters=[96, 96, 128], stage=3, block='voice_2a', strides=(2, 2), kernel_initializer=kernel_initializer)
        x2 = self.identity_block2d(x2, kernel_size=3, filters=[96, 96, 128], stage=3, block='voice_2b', kernel_initializer=kernel_initializer)
        x2 = self.identity_block2d(x2, kernel_size=3, filters=[96, 96, 128], stage=3, block='voice_2c', kernel_initializer=kernel_initializer)

        # Conv 4_x
        x3 = self.conv_block_2d(x2, kernel_size=3, filters=[128, 128, 256], stage=4, block='voice_3a', strides=(2, 2), kernel_initializer=kernel_initializer)
        x3 = self.identity_block2d(x3, kernel_size=3, filters=[128, 128, 256], stage=4, block='voice_3b', kernel_initializer=kernel_initializer)
        x3 = self.identity_block2d(x3, kernel_size=3, filters=[128, 128, 256], stage=4, block='voice_3c', kernel_initializer=kernel_initializer)

        # Conv 5_x
        x4 = self.conv_block_2d(x3, kernel_size=3, filters=[256, 256, 512], stage=5, block='voice_4a', strides=(2, 2), kernel_initializer=kernel_initializer)
        x4 = self.identity_block2d(x4, kernel_size=3, filters=[256, 256, 512], stage=5, block='voice_4b', kernel_initializer=kernel_initializer)
        x4 = self.identity_block2d(x4, kernel_size=3, filters=[256, 256, 512], stage=5, block='voice_4c', kernel_initializer=kernel_initializer)

        # Fc layers
        embedding_layer = self.__conv_bn_dynamic_apool(x4, layer_idx=6, conv_filters=512, conv_kernel_size=(9, 1), conv_strides=(1, 1), conv_pad=(0, 0), conv_layer_prefix='fc')
        x = tf.keras.layers.Lambda(lambda y: tf.keras.backend.l2_normalize(y, axis=3), name='norm')(embedding_layer)
        output = tf.keras.layers.Conv2D(filters=classes, kernel_size=(1, 1), strides=(1, 1), padding='valid', name='fc8')(x)

        self.model = tf.keras.Model(inputs=[signal_input, impulse_input], outputs=[output])
        print('>', 'built', self.name, 'model on', classes, 'classes')

        super().load()

        self.inference_model = tf.keras.Model(inputs=[signal_input, impulse_input], outputs=[embedding_layer])
        print('>', 'built', self.name, 'inference model')

        self.model.summary()