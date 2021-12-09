#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import os

from models.verifier.model import VladPooling
from models.verifier.model import Model

from helpers.audio import get_tf_spectrum

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class ResNet34(Model):

    def __init__(self, name='resnet34', id=''):
        super().__init__(name, id)

    def compute_acoustic_representation(self, e):
        return get_tf_spectrum(e, num_fft=512)

    def identity_block_2d(self, input_tensor, kernel_size, filters, stage, block, weight_decay, trainable):
        filters1, filters2, filters3 = filters

        x = tf.keras.layers.Conv2D(filters1, (1, 1), kernel_initializer='orthogonal', use_bias=False, trainable=trainable, kernel_regularizer= tf.keras.regularizers.l2(weight_decay), name='conv' + str(stage) + '_' + str(block) + '_1x1_reduce')(input_tensor)
        x = tf.keras.layers.BatchNormalization(axis=3, trainable=trainable, name='conv' + str(stage) + '_' + str(block) + '_1x1_reduce/bn')(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same', kernel_initializer='orthogonal', use_bias=False, trainable=trainable, kernel_regularizer=tf.keras.regularizers.l2(weight_decay), name='conv' + str(stage) + '_' + str(block) + '_3x3')(x)
        x = tf.keras.layers.BatchNormalization(axis=3, trainable=trainable, name='conv' + str(stage) + '_' + str(block) + '_3x3/bn')(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Conv2D(filters3, (1, 1), kernel_initializer='orthogonal', use_bias=False, trainable=trainable, kernel_regularizer=tf.keras.regularizers.l2(weight_decay), name='conv' + str(stage) + '_' + str(block) + '_1x1_increase')(x)
        x = tf.keras.layers.BatchNormalization(axis=3, trainable=trainable, name='conv' + str(stage) + '_' + str(block) + '_1x1_increase/bn')(x)

        x = tf.keras.layers.Add()([x, input_tensor])
        x = tf.keras.layers.ReLU()(x)
        return x

    def conv_block_2d(self, input_tensor, kernel_size, filters, stage, block, strides, weight_decay, trainable):
        filters1, filters2, filters3 = filters

        bn_name_1 = 'conv' + str(stage) + '_' + str(block) + '_1x1_reduce/bn'
        x = tf.keras.layers.Conv2D(filters1, (1, 1), strides=strides, kernel_initializer='orthogonal', use_bias=False, trainable=trainable, kernel_regularizer=tf.keras.regularizers.l2(weight_decay), name='conv' + str(stage) + '_' + str(block) + '_1x1_reduce')(input_tensor)
        x = tf.keras.layers.BatchNormalization(axis=3, trainable=trainable, name=bn_name_1)(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same', kernel_initializer='orthogonal', use_bias=False, trainable=trainable, kernel_regularizer=tf.keras.regularizers.l2(weight_decay), name='conv' + str(stage) + '_' + str(block) + '_3x3')(x)
        x = tf.keras.layers.BatchNormalization(axis=3, trainable=trainable, name='conv' + str(stage) + '_' + str(block) + '_3x3/bn')(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Conv2D(filters3, (1, 1), kernel_initializer='orthogonal', use_bias=False, trainable=trainable, kernel_regularizer=tf.keras.regularizers.l2(weight_decay), name='conv' + str(stage) + '_' + str(block) + '_1x1_increase')(x)
        x = tf.keras.layers.BatchNormalization(axis=3, trainable=trainable, name='conv' + str(stage) + '_' + str(block) + '_1x1_increase/bn')(x)

        shortcut = tf.keras.layers.Conv2D(filters3, (1, 1), strides=strides, kernel_initializer='orthogonal', use_bias=False, trainable=trainable, kernel_regularizer=tf.keras.regularizers.l2(weight_decay), name='conv' + str(stage) + '_' + str(block) + '_1x1_proj')(input_tensor)
        shortcut = tf.keras.layers.BatchNormalization(axis=3, trainable=trainable, name='conv' + str(stage) + '_' + str(block) + '_1x1_proj/bn')(shortcut)

        x = tf.keras.layers.Add()([x, shortcut])
        x = tf.keras.layers.ReLU()(x)

        return x


    def build(self, classes=0, embs_name='embs', embs_size=512, loss='softmax', aggregation='gvlad', vlad_clusters=10, ghost_clusters=2, weight_decay=1e-3, mode='train'):
        super().build(classes, embs_name, embs_size, loss, aggregation, vlad_clusters, ghost_clusters, weight_decay, mode)
        print('>', 'building', self.name, 'model on', classes, 'classes')

        input_layer = tf.keras.Input(shape=(256, None, 1,), name='Input_1')

        x1 = tf.keras.layers.Conv2D(64, (7, 7), kernel_initializer='orthogonal', use_bias=False, trainable=True, kernel_regularizer=tf.keras.regularizers.l2(weight_decay), padding='same', name='conv1_1/3x3_s1')(input_layer)
        x1 = tf.keras.layers.BatchNormalization(axis=3, name='conv1_1/3x3_s1/bn', trainable=True)(x1)
        x1 = tf.keras.layers.ReLU()(x1)
        x1 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x1)

        # Conv 2
        x2 = self.conv_block_2d(x1, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), weight_decay=weight_decay, trainable=True)
        x2 = self.identity_block_2d(x2, 3, [64, 64, 256], stage=2, block='b', weight_decay=weight_decay, trainable=True)
        x2 = self.identity_block_2d(x2, 3, [64, 64, 256], stage=2, block='c', weight_decay=weight_decay, trainable=True)

        # Conv 3
        x3 = self.conv_block_2d(x2, 3, [128, 128, 512], stage=3, block='a', strides=(2, 2), weight_decay=weight_decay, trainable=True)
        x3 = self.identity_block_2d(x3, 3, [128, 128, 512], stage=3, block='b', weight_decay=weight_decay, trainable=True)
        x3 = self.identity_block_2d(x3, 3, [128, 128, 512], stage=3, block='c', weight_decay=weight_decay, trainable=True)

        # Conv 4
        x4 = self.conv_block_2d(x3, 3, [256, 256, 1024], stage=4, block='a', strides=(1, 1), weight_decay=weight_decay, trainable=True)
        x4 = self.identity_block_2d(x4, 3, [256, 256, 1024], stage=4, block='b', weight_decay=weight_decay, trainable=True)
        x4 = self.identity_block_2d(x4, 3, [256, 256, 1024], stage=4, block='c', weight_decay=weight_decay, trainable=True)

        # Conv 5
        x5 = self.conv_block_2d(x4, 3, [512, 512, 2048], stage=5, block='a', strides=(2, 2), weight_decay=weight_decay, trainable=True)
        x5 = self.identity_block_2d(x5, 3, [512, 512, 2048], stage=5, block='b', weight_decay=weight_decay, trainable=True)
        x5 = self.identity_block_2d(x5, 3, [512, 512, 2048], stage=5, block='c', weight_decay=weight_decay, trainable=True)
        x = tf.keras.layers.MaxPooling2D((3, 1), strides=(2, 1), name='mpool2')(x5)

        # Fc layers
        x_fc = tf.keras.layers.Conv2D(embs_size, (7, 1), strides=(1, 1), kernel_initializer='orthogonal', use_bias=False, trainable=True, kernel_regularizer=tf.keras.regularizers.l2(weight_decay), name='x_fc')(x)

        if aggregation == 'avg':
            if mode == 'train':
                x = tf.keras.layers.AveragePooling2D((1, 5), strides=(1, 1), name='avg_pool')(x)
                x = tf.keras.layers.Reshape((-1, embs_size))(x)
            else:
                x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
                x = tf.keras.layers.Reshape((1, embs_size))(x)

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
        self.model = tf.keras.Model(inputs=input_layer, outputs=output_layer, name='resnet34_{}_{}'.format(loss, aggregation))

        print('>', 'built', self.name, 'model on', classes, 'classes')