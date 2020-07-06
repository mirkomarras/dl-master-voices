#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import os

from models.verifier.model import VladPooling
from models.verifier.model import Model

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

    def build(self, classes=None, loss='softmax', aggregation='avg', vlad_clusters=12, ghost_clusters=2, weight_decay=1e-4, training_phase=True):
        super().build(classes, loss, aggregation, vlad_clusters, ghost_clusters, weight_decay)
        print('>', 'building', self.name, 'model on', classes, 'classes')

        input_layer = tf.keras.Input(shape=(512,None,1,), name='Input_1')

        resnet_50 = tf.keras.applications.ResNet50(input_tensor=input_layer, include_top=False, weights=None)

        x = tf.keras.layers.ZeroPadding2D(padding=(0, 0), name='pad{}'.format(6))(resnet_50.output)
        xfc = tf.keras.layers.Conv2D(filters=self.emb_size, kernel_size=(9, 1), strides=(1, 1), padding='valid', kernel_initializer='orthogonal', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(weight_decay), name='{}{}'.format('fc', 6))(x)
        xfc = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=1., trainable=training_phase, name='bn{}'.format(6))(xfc)
        xfc = tf.keras.layers.Activation('relu', name='relu{}'.format(6))(xfc)

        if aggregation == 'avg':
            x = tf.keras.layers.AveragePooling2D(pool_size=(1, 8), strides=(1, 1), name='apool{}'.format(6))(xfc)
            x = tf.math.reduce_mean(x, axis=[1, 2], name='rmean{}'.format(6))
            x = tf.keras.layers.Lambda(lambda x: tf.keras.backend.l2_normalize(x, 1))(x)
        elif aggregation == 'vlad':
            xkcenter = tf.keras.layers.Conv2D(vlad_clusters + ghost_clusters, (9, 1), strides=(1, 1), kernel_initializer='orthogonal', use_bias=True, kernel_regularizer=tf.keras.regularizers.l2(weight_decay), bias_regularizer=tf.keras.regularizers.l2(weight_decay), name='vlad_center_assignment')(x)
            x = VladPooling(k_centers=vlad_clusters, g_centers=ghost_clusters, mode='vlad', name='vlad_pool')([xfc, xkcenter])
        elif aggregation == 'gvlad':
            xkcenter = tf.keras.layers.Conv2D(vlad_clusters + ghost_clusters, (9, 1), strides=(1, 1), kernel_initializer='orthogonal', use_bias=True, kernel_regularizer=tf.keras.regularizers.l2(weight_decay), bias_regularizer=tf.keras.regularizers.l2(weight_decay), name='gvlad_center_assignment')(x)
            x = VladPooling(k_centers=vlad_clusters, g_centers=ghost_clusters, mode='gvlad', name='gvlad_pool')([xfc, xkcenter])
        else:
            raise NotImplementedError()

        x = tf.keras.layers.Lambda(lambda x: tf.keras.backend.l2_normalize(x, 1))(x)
        e = tf.keras.layers.Dense(self.emb_size, activation='relu', kernel_initializer='orthogonal', use_bias=True, kernel_regularizer=tf.keras.regularizers.l2(weight_decay), bias_regularizer=tf.keras.regularizers.l2(weight_decay), name='fc7')(x)

        if loss == 'softmax':
            y = tf.keras.layers.Dense(classes, activation='softmax', kernel_initializer='orthogonal', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(weight_decay), bias_regularizer=tf.keras.regularizers.l2(weight_decay), name='fc8')(e)
        elif loss == 'amsoftmax':
            x = tf.keras.layers.Lambda(lambda x: tf.keras.backend.l2_normalize(x, 1))(e)
            y = tf.keras.layers.Dense(classes, kernel_initializer='orthogonal', use_bias=False, kernel_constraint=tf.keras.constraints.unit_norm(), kernel_regularizer=tf.keras.regularizers.l2(weight_decay), bias_regularizer=tf.keras.regularizers.l2(weight_decay), name='fc8')(x)
        else:
            raise NotImplementedError()

        self.model = tf.keras.Model(inputs=input_layer, outputs=y)
        print('>', 'built', self.name, 'model on', classes, 'classes')
