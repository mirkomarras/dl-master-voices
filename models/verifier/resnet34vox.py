#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import os

from models.verifier.model import VladPooling
from models.verifier.model import Model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class ResNet34Vox(Model):

    def __init__(self, name='resnet34vox', id='', noises=None, cache=None, n_seconds=3, sample_rate=16000):
        super().__init__(name, id, noises, cache, n_seconds, sample_rate)

    def identity_block2d(self, input_tensor, kernel_size, filters, stage, block, weight_decay):
        filters1, filters2, filters3 = filters
        conv_name_1 = 'conv' + str(stage) + '_' + str(block) + '_1x1_reduce'
        bn_name_1 = 'bn' + str(stage) + '_' + str(block) + '_1x1_reduce'
        x = tf.keras.layers.Conv2D(filters=filters1, kernel_size=(1, 1), kernel_initializer='orthogonal', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(weight_decay), name=conv_name_1)(input_tensor)
        x = tf.keras.layers.BatchNormalization(name=bn_name_1)(x)
        x = tf.keras.layers.ReLU()(x)
        conv_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3'
        bn_name_2 = 'bn' + str(stage) + '_' + str(block) + '_3x3'
        x = tf.keras.layers.Conv2D(filters=filters2, kernel_size=kernel_size, padding='SAME', kernel_initializer='orthogonal', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(weight_decay), name=conv_name_2)(x)
        x = tf.keras.layers.BatchNormalization(name=bn_name_2)(x)
        x = tf.keras.layers.ReLU()(x)
        conv_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase'
        bn_name_3 = 'bn' + str(stage) + '_' + str(block) + '_1x1_increase'
        x = tf.keras.layers.Conv2D(filters=filters3, kernel_size=(1, 1), kernel_initializer='orthogonal', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(weight_decay), name=conv_name_3)(x)
        x = tf.keras.layers.BatchNormalization(name=bn_name_3)(x)
        x = tf.keras.layers.Add()([input_tensor, x])
        x = tf.keras.layers.ReLU()(x)
        return x

    def conv_block_2d(self, input_tensor, kernel_size, filters, stage, block, strides, weight_decay):
        filters1, filters2, filters3 = filters
        conv_name_1 = 'conv' + str(stage) + '_' + str(block) + '_1x1_reduce'
        bn_name_1 = 'bn' + str(stage) + '_' + str(block) + '_1x1_reduce'
        x = tf.keras.layers.Conv2D(filters=filters1, kernel_size=(1, 1), strides=strides, kernel_initializer='orthogonal', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(weight_decay), name=conv_name_1)(input_tensor)
        x = tf.keras.layers.BatchNormalization(name=bn_name_1)(x)
        x = tf.keras.layers.ReLU()(x)
        conv_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3'
        bn_name_2 = 'bn' + str(stage) + '_' + str(block) + '_3x3'
        x = tf.keras.layers.Conv2D(filters=filters2, kernel_size=kernel_size, padding='SAME', kernel_initializer='orthogonal', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(weight_decay), name=conv_name_2)(x)
        x = tf.keras.layers.BatchNormalization(name=bn_name_2)(x)
        x = tf.keras.layers.ReLU()(x)
        conv_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase'
        bn_name_3 = 'bn' + str(stage) + '_' + str(block) + '_1x1_increase'
        x = tf.keras.layers.Conv2D(filters=filters3, kernel_size=(1, 1), kernel_initializer='orthogonal', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(weight_decay), name=conv_name_3)(x)
        x = tf.keras.layers.BatchNormalization(name=bn_name_3)(x)
        conv_name_4 = 'conv' + str(stage) + '_' + str(block) + '_1x1_shortcut'
        bn_name_4 = 'bn' + str(stage) + '_' + str(block) + '_1x1_shortcut'
        shortcut = tf.keras.layers.Conv2D(filters=filters3, kernel_size=(1, 1), strides=strides, kernel_initializer='orthogonal', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(weight_decay), name=conv_name_4)(input_tensor)
        shortcut = tf.keras.layers.BatchNormalization(name=bn_name_4)(shortcut)
        x = tf.keras.layers.Add()([shortcut, x])
        x = tf.keras.layers.ReLU()(x)
        return x

    def build(self, classes=None, loss='softmax', aggregation='avg', vlad_clusters=12, ghost_clusters=2, weight_decay=1e-4):
        super().build(classes, loss, aggregation, vlad_clusters, ghost_clusters, weight_decay)
        print('>', 'building', self.name, 'model on', classes, 'classes')

        input_layer = tf.keras.Input(shape=(512,None,1,), name='Input_1')

        # Conv 1
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), kernel_initializer='orthogonal', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(weight_decay), padding='SAME', name='voice_conv1_1/3x3_s1')(input_layer)
        x = tf.keras.layers.BatchNormalization(name='voice_bn1_1/3x3_s1')(x)
        x = tf.keras.layers.ReLU()(x)

        # Pool 1
        x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='voice_mpool1')(x)

        # Conv 2_x
        x1 = self.conv_block_2d(x, 3, filters=[48, 48, 96], stage=2, block='voice_1a', strides=(1, 1), weight_decay=weight_decay)
        x1 = self.identity_block2d(x1, 3, filters=[48, 48, 96], stage=2, block='voice_1b', weight_decay=weight_decay)

        # Conv 3_x
        x2 = self.conv_block_2d(x1, 3, filters=[96, 96, 128], stage=3, block='voice_2a', strides=(2, 2), weight_decay=weight_decay)
        x2 = self.identity_block2d(x2, 3, filters=[96, 96, 128], stage=3, block='voice_2b', weight_decay=weight_decay)
        x2 = self.identity_block2d(x2, 3, filters=[96, 96, 128], stage=3, block='voice_2c', weight_decay=weight_decay)

        # Conv 4_x
        x3 = self.conv_block_2d(x2, 3, filters=[128, 128, 256], stage=4, block='voice_3a', strides=(2, 2), weight_decay=weight_decay)
        x3 = self.identity_block2d(x3, 3, filters=[128, 128, 256], stage=4, block='voice_3b', weight_decay=weight_decay)
        x3 = self.identity_block2d(x3, 3, filters=[128, 128, 256], stage=4, block='voice_3c', weight_decay=weight_decay)

        # Conv 5_x
        x4 = self.conv_block_2d(x3, 3, filters=[256, 256, 512], stage=5, block='voice_4a', strides=(2, 2), weight_decay=weight_decay)
        x4 = self.identity_block2d(x4, 3, filters=[256, 256, 512], stage=5, block='voice_4b', weight_decay=weight_decay)
        x4 = self.identity_block2d(x4, 3, filters=[256, 256, 512], stage=5, block='voice_4c', weight_decay=weight_decay)
        x = tf.keras.layers.MaxPooling2D((3, 1), strides=(2, 1), name='mpool2')(x4)

        # Fc layers
        xfc = tf.keras.layers.Conv2D(filters=self.emb_size, kernel_size=(7, 1), strides=(1, 1), padding='valid', kernel_initializer='orthogonal', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(weight_decay), name='{}{}'.format('fc', 6))(x)

        if aggregation == 'avg':
            x = tf.keras.layers.AveragePooling2D(pool_size=(1, 8), strides=(1, 1), name='apool{}'.format(6))(x)
            x = tf.math.reduce_mean(x, axis=[1, 2], name='rmean{}'.format(6))
            x = tf.keras.layers.Lambda(lambda x: tf.keras.backend.l2_normalize(x, 1))(x)
        elif aggregation == 'vlad':
            xkcenter = tf.keras.layers.Conv2D(vlad_clusters + ghost_clusters, (7, 1), strides=(1, 1), kernel_initializer='orthogonal', use_bias=True, kernel_regularizer=tf.keras.regularizers.l2(weight_decay), bias_regularizer=tf.keras.regularizers.l2(weight_decay), name='vlad_center_assignment')(x)
            x = VladPooling(k_centers=vlad_clusters, g_centers=ghost_clusters, mode='vlad', name='vlad_pool')([xfc, xkcenter])
        elif aggregation == 'gvlad':
            xkcenter = tf.keras.layers.Conv2D(vlad_clusters + ghost_clusters, (7, 1), strides=(1, 1), kernel_initializer='orthogonal', use_bias=True, kernel_regularizer=tf.keras.regularizers.l2(weight_decay), bias_regularizer=tf.keras.regularizers.l2(weight_decay), name='gvlad_center_assignment')(x)
            x = VladPooling(k_centers=vlad_clusters, g_centers=ghost_clusters, mode='gvlad', name='gvlad_pool')([xfc, xkcenter])
        else:
            raise NotImplementedError()

        x = tf.keras.layers.Dense(self.emb_size, activation='relu', kernel_initializer='orthogonal', use_bias=True, kernel_regularizer=tf.keras.regularizers.l2(weight_decay), bias_regularizer=tf.keras.regularizers.l2(weight_decay), name='fc7')(x)

        if loss == 'softmax':
            y = tf.keras.layers.Dense(classes, activation='softmax', kernel_initializer='orthogonal', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(weight_decay), bias_regularizer=tf.keras.regularizers.l2(weight_decay), name='fc8')(x)
        elif loss == 'amsoftmax':
            x = tf.keras.layers.Lambda(lambda x: tf.keras.backend.l2_normalize(x, 1))(x)
            y = tf.keras.layers.Dense(classes, kernel_initializer='orthogonal', use_bias=False, kernel_constraint=tf.keras.constraints.unit_norm(), kernel_regularizer=tf.keras.regularizers.l2(weight_decay), bias_regularizer=tf.keras.regularizers.l2(weight_decay), name='fc8')(x)
        else:
            raise NotImplementedError()

        self.model = tf.keras.Model(inputs=input_layer, outputs=y)
        print('>', 'built', self.name, 'model on', classes, 'classes')