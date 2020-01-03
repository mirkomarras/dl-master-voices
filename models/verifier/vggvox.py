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

    def __conv_bn_pool(self, inp_tensor, layer_idx, conv_filters, conv_kernel_size, conv_strides, conv_pad, pool='', pool_size=(2, 2), pool_strides=None, conv_layer_prefix='conv'):
        x = tf.keras.layers.ZeroPadding2D(padding=conv_pad, name='pad{}'.format(layer_idx))(inp_tensor)
        x = tf.keras.layers.Conv2D(filters=conv_filters, kernel_size=conv_kernel_size, strides=conv_strides, padding='valid', name='{}{}'.format(conv_layer_prefix, layer_idx))(x)
        x = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=1., name='bn{}'.format(layer_idx))(x)
        x = tf.keras.layers.Activation('relu', name='relu{}'.format(layer_idx))(x)
        if pool == 'max':
            x = tf.keras.layers.MaxPooling2D(pool_size=pool_size, strides=pool_strides, name='mpool{}'.format(layer_idx))(x)
        elif pool == 'avg':
            x = tf.keras.layers.AveragePooling2D(pool_size=pool_size, strides=pool_strides, name='apool{}'.format(layer_idx))(x)
        return x

    def __conv_bn_dynamic_apool(self, inp_tensor, layer_idx, conv_filters, conv_kernel_size, conv_strides, conv_pad, conv_layer_prefix='conv'):
        x = tf.keras.layers.ZeroPadding2D(padding=conv_pad, name='pad{}'.format(layer_idx))(inp_tensor)
        x = tf.keras.layers.Conv2D(filters=conv_filters, kernel_size=conv_kernel_size, strides=conv_strides, padding='valid', name='{}{}'.format(conv_layer_prefix, layer_idx))(x)
        x = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=1., name='bn{}'.format(layer_idx))(x)
        x = tf.keras.layers.Activation('relu', name='relu{}'.format(layer_idx))(x)
        x = tf.keras.layers.AveragePooling2D(pool_size=(1, 8), strides=(1, 1), name='apool{}'.format(layer_idx))(x)
        x = tf.keras.layers.Reshape((1, 1, conv_filters), name='reshape{}'.format(layer_idx))(x)
        return x
    
    def build(self, classes=None):
        super().build(classes)
        print('>', 'building', self.name, 'model on', classes, 'classes')

        signal_input = tf.keras.Input(shape=(None,1,))
        impulse_input = tf.keras.Input(shape=(3,))

        x = tf.keras.layers.Lambda(lambda x: play_n_rec(x, self.noises, self.cache), name='play_n_rec')([signal_input, impulse_input])
        x = tf.keras.layers.Lambda(lambda x: get_tf_spectrum(x, self.sample_rate), name='acoustic_layer')(x)

        x = self.__conv_bn_pool(x, layer_idx=1, conv_filters=96, conv_kernel_size=(7, 7), conv_strides=(2, 2), conv_pad=(1, 1), pool='max', pool_size=(3, 3), pool_strides=(2, 2))
        x = self.__conv_bn_pool(x, layer_idx=2, conv_filters=256, conv_kernel_size=(5, 5), conv_strides=(2, 2), conv_pad=(1, 1), pool='max', pool_size=(3, 3), pool_strides=(2, 2))
        x = self.__conv_bn_pool(x, layer_idx=3, conv_filters=384, conv_kernel_size=(3, 3), conv_strides=(1, 1), conv_pad=(1, 1))
        x = self.__conv_bn_pool(x, layer_idx=4, conv_filters=256, conv_kernel_size=(3, 3), conv_strides=(1, 1), conv_pad=(1, 1))
        x = self.__conv_bn_pool(x, layer_idx=5, conv_filters=256, conv_kernel_size=(3, 3), conv_strides=(1, 1), conv_pad=(1, 1), pool='max', pool_size=(5, 3), pool_strides=(3, 2))
        x = self.__conv_bn_dynamic_apool(x, layer_idx=6, conv_filters=4096, conv_kernel_size=(9, 1), conv_strides=(1, 1), conv_pad=(0, 0), conv_layer_prefix='fc')
        embedding_layer = self.__conv_bn_pool(x, layer_idx=7, conv_filters=1024, conv_kernel_size=(1, 1), conv_strides=(1, 1), conv_pad=(0, 0),  conv_layer_prefix='fc')
        output = tf.keras.layers.Conv2D(filters=classes, kernel_size=(1, 1), strides=(1, 1), padding='valid', name='fc8')(embedding_layer)

        self.model = tf.keras.Model(inputs=[signal_input, impulse_input], outputs=[output])
        print('>', 'built', self.name, 'model on', classes, 'classes')

        super().load()

        self.inference_model = tf.keras.Model(inputs=[signal_input, impulse_input], outputs=[embedding_layer])
        print('>', 'built', self.name, 'inference model')