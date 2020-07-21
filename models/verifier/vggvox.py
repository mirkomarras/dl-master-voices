#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import os

from models.verifier.model import VladPooling
from models.verifier.model import Model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class VggVox(Model):

    """
       Class to represent Speaker Verification (SV) model based on the VGG16 architecture - Embedding vectors of size 1024 are returned
       Nagrani, A., Chung, J. S., & Zisserman, A. (2017).
       VoxCeleb: A Large-Scale Speaker Identification Dataset.
       Proc. Interspeech 2017, 2616-2620.
    """

    def __init__(self, name='vggvox', id=-1):
        super().__init__(name, id)

    def __conv_bn_pool(self, x, layer_idx, conv_filters, conv_kernel_size, conv_strides, pool='', pool_size=(2, 2), pool_strides=None):
        x = tf.keras.layers.Conv2D(filters=conv_filters, kernel_size=conv_kernel_size, strides=conv_strides, padding='valid', name='{}{}'.format('conv', layer_idx))(x)
        x = tf.keras.layers.BatchNormalization(name='bn{}'.format(layer_idx))(x)
        x = tf.keras.layers.Activation('relu', name='relu{}'.format(layer_idx))(x)
        if pool == 'max':
            x = tf.keras.layers.MaxPooling2D(pool_size=pool_size, strides=pool_strides, name='mpool{}'.format(layer_idx))(x)
        elif pool == 'avg':
            x = tf.keras.layers.AveragePooling2D(pool_size=pool_size, strides=pool_strides, name='apool{}'.format(layer_idx))(x)
        return x

    def build(self, classes=None, embs_size=512, embs_name='embs', loss='softmax', aggregation='avg', vlad_clusters=12, ghost_clusters=2, weight_decay=1e-3, mode='train'):
        super().build(classes, embs_size, embs_name, loss, aggregation, vlad_clusters, ghost_clusters, weight_decay, mode)
        print('>', 'building', self.name, 'model on', classes, 'classes')

        input_layer = tf.keras.Input(shape=(256, None, 1,), name='Input_1')

        x = self.__conv_bn_pool(input_layer, layer_idx=1, conv_filters=96, conv_kernel_size=(7, 7), conv_strides=(2, 2), pool='max', pool_size=(3, 3), pool_strides=(2, 2))
        x = self.__conv_bn_pool(x, layer_idx=2, conv_filters=256, conv_kernel_size=(5, 5), conv_strides=(2, 2), pool='max', pool_size=(3, 3), pool_strides=(2, 2))
        x = self.__conv_bn_pool(x, layer_idx=3, conv_filters=384, conv_kernel_size=(3, 3), conv_strides=(1, 1))
        x = self.__conv_bn_pool(x, layer_idx=4, conv_filters=256, conv_kernel_size=(3, 3), conv_strides=(1, 1))
        x = self.__conv_bn_pool(x, layer_idx=5, conv_filters=256, conv_kernel_size=(3, 3), conv_strides=(1, 1), pool='max', pool_size=(5, 3), pool_strides=(3, 2))

        # Fc layers
        x_fc = tf.keras.layers.Conv2D(embs_size, (2, 1), strides=(1, 1), kernel_initializer='orthogonal', use_bias=False, trainable=True, kernel_regularizer=tf.keras.regularizers.l2(weight_decay), name='x_fc')(x)

        if aggregation == 'avg':
            if mode == 'train':
                x = tf.keras.layers.AveragePooling2D((1, 5), strides=(1, 1), name='avg_pool')(x)
                x = tf.keras.layers.Reshape((-1, embs_size))(x)
            else:
                x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
                x = tf.keras.layers.Reshape((1, embs_size))(x)

        elif aggregation == 'vlad':
            x_k_center = tf.keras.layers.Conv2D(vlad_clusters, (2, 1), strides=(1, 1), kernel_initializer='orthogonal', use_bias=True, trainable=True, kernel_regularizer=tf.keras.regularizers.l2(weight_decay), bias_regularizer=tf.keras.regularizers.l2(weight_decay), name='vlad_center_assignment')(x)
            x = VladPooling(k_centers=vlad_clusters, mode='vlad', name='vlad_pool')([x_fc, x_k_center])

        elif aggregation == 'gvlad':
            x_k_center = tf.keras.layers.Conv2D(vlad_clusters + ghost_clusters, (2, 1), strides=(1, 1), kernel_initializer='orthogonal', use_bias=True, trainable=True, kernel_regularizer=tf.keras.regularizers.l2(weight_decay), bias_regularizer=tf.keras.regularizers.l2(weight_decay), name='gvlad_center_assignment')(x)
            x = VladPooling(k_centers=vlad_clusters, g_centers=ghost_clusters, mode='gvlad', name='gvlad_pool')([x_fc, x_k_center])

        else:
            raise NotImplementedError()

        x = tf.keras.layers.Dense(embs_size, activation='relu', kernel_initializer='orthogonal', use_bias=True, trainable=True, kernel_regularizer=tf.keras.regularizers.l2(weight_decay), bias_regularizer=tf.keras.regularizers.l2(weight_decay), name='embs')(x)

        if loss == 'softmax':
            output_layer = tf.keras.layers.Dense(classes, activation='softmax', kernel_initializer='orthogonal', use_bias=False, trainable=True, kernel_regularizer=tf.keras.regularizers.l2(weight_decay), bias_regularizer=tf.keras.regularizers.l2(weight_decay), name='prediction')(x)

        elif loss == 'amsoftmax':
            x_l2 = tf.keras.layers.Lambda(lambda x: tf.keras.backend.l2_normalize(x, 1))(x)
            output_layer = tf.keras.layers.Dense(classes, kernel_initializer='orthogonal', use_bias=False, trainable=True, kernel_constraint=tf.keras.constraints.unit_norm(), kernel_regularizer=tf.keras.regularizers.l2(weight_decay), bias_regularizer=tf.keras.regularizers.l2(weight_decay), name='prediction')(x_l2)

        else:
            raise NotImplementedError()

        self.embs_name = embs_name
        self.model = tf.keras.Model(inputs=input_layer, outputs=output_layer, name='vggvox_{}_{}'.format(loss, aggregation))

        print('>', 'built', self.name, 'model on', classes, 'classes')