#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import os

from models.verifier.model import VladPooling
from models.verifier.model import Model

from helpers.audio import get_tf_spectrum

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class ResNet50(Model):

    """
       Class to represent Speaker Verification (SV) model based on the ResNet50 architecture - Embedding vectors of size 512 are returned
       Chung, J. S., Nagrani, A., & Zisserman, A. (2018).
       VoxCeleb2: Deep Speaker Recognition.
       Proc. Interspeech 2018, 1086-1090.
    """

    def __init__(self, name='resnet50', id=''):
        super().__init__(name, id)


    def compute_acoustic_representation(self, e):
        return get_tf_spectrum(e, num_fft=512)


    def build(self, classes=0, embs_name='embs', embs_size=512, loss='softmax', aggregation='gvlad', vlad_clusters=12, ghost_clusters=2, weight_decay=1e-3, mode='train'):
        super().build(classes, embs_name, embs_size, loss, aggregation, vlad_clusters, ghost_clusters, weight_decay, mode)
        print('>', 'building', self.name, 'model on', classes, 'classes')

        input_layer = tf.keras.Input(shape=(256, None, 1,), name='Input_1')

        backbone = tf.keras.applications.ResNet50(input_tensor=input_layer, include_top=False, weights=None)

        x = backbone.output

        # Fc layers
        x_fc = tf.keras.layers.Conv2D(embs_size, (7, 1), strides=(1, 1), kernel_initializer='orthogonal', use_bias=False, trainable=True, kernel_regularizer=tf.keras.regularizers.l2(weight_decay), name='x_fc')(x)

        if aggregation == 'avg':
            if mode == 'train':
                x = tf.keras.layers.AveragePooling2D((1, 5), strides=(1, 1), name='avg_pool')(x)
                x = tf.keras.layers.Reshape((-1, embs_size))(x)
            else:
                x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
                x = tf.keras.layers.Reshape((-1, embs_size))(x)

        elif aggregation == 'vlad':
            x_k_center = tf.keras.layers.Conv2D(vlad_clusters, (7, 1), strides=(1, 1), kernel_initializer='orthogonal', use_bias=True, trainable=True, kernel_regularizer=tf.keras.regularizers.l2(weight_decay), bias_regularizer=tf.keras.regularizers.l2(weight_decay), name='vlad_center_assignment')(x)
            x = VladPooling(k_centers=vlad_clusters, mode='vlad', name='vlad_pool')([x_fc, x_k_center])

        elif aggregation == 'gvlad':
            x_k_center = tf.keras.layers.Conv2D(vlad_clusters + ghost_clusters, (7, 1), strides=(1, 1), kernel_initializer='orthogonal', use_bias=True, trainable=True, kernel_regularizer=tf.keras.regularizers.l2(weight_decay), bias_regularizer=tf.keras.regularizers.l2(weight_decay), name='gvlad_center_assignment')(x)
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
        self.model = tf.keras.Model(inputs=input_layer, outputs=output_layer, name='resnet50_{}_{}'.format(loss, aggregation))

        print('>', 'built', self.name, 'model on', classes, 'classes')
