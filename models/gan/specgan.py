#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import os

from models.gan.model import GAN
from helpers.audio import get_tf_spectrum

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class SpecGAN(GAN):

    def __init__(self, name='specgan', id=-1, gender='neutral', latent_dim=100, slice_len=128):
        super().__init__(name, id, gender, latent_dim)
        self.slice_len = slice_len
        self.kernel_size = 25
        self.gan_dim = 64
        self.upsample = 'zeros'
        self.stride = 2 if self.slice_len == 32768 else 4
        self.is_raw = False

    def __conv2d_transpose(self, inputs, filters):
        if self.upsample == 'zeros':
            return tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=(1, self.kernel_size), strides=(self.stride, self.stride), padding='SAME')(tf.expand_dims(inputs, axis=1))[:, 0]
        else:
            _, h, w, nch = inputs.get_shape().as_list()
            x = inputs
            if upsample == 'nn':
                x = tf.image.resize_nearest_neighbor(x, [h * self.stride, w * self.stride])
            elif upsample == 'linear':
                x = tf.image.resize_bilinear(x, [h * self.stride, w * self.stride])
            else:
                x = tf.image.resize_bicubic(x, [h * self.stride, w * self.stride])
            return tf.keras.layers.Conv2D(filters, self.kernel_size, (1,1), padding='SAME', dtype='float32')(x)

    def build_discriminator_model(self):
        input = tf.keras.Input((self.slice_len,1,), dtype='float32')

        x = tf.pad(input, [[0, 0], [0, 128], [0, 0]], 'CONSTANT')
        x = tf.keras.layers.Lambda(lambda x: get_tf_spectrum(x, frame_size=0.016, frame_stride=0.008, num_fft=256))(x)

        x = tf.keras.layers.Conv2D(self.gan_dim, self.kernel_size, 2, padding='SAME')(x)
        x = tf.maximum(0.2 * x, x)

        x = tf.keras.layers.Conv2D(self.gan_dim * 2, self.kernel_size, 2, padding='SAME')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.maximum(0.2 * x, x)

        x = tf.keras.layers.Conv2D(self.gan_dim * 4, self.kernel_size, 2, padding='SAME')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.maximum(0.2 * x, x)

        x = tf.keras.layers.Conv2D(self.gan_dim * 8, self.kernel_size, 2, padding='SAME')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.maximum(0.2 * x, x)

        x = tf.keras.layers.Conv2D(self.gan_dim * 16, self.kernel_size, 2, padding='SAME')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.maximum(0.2 * x, x)

        x = tf.keras.layers.Reshape([4 * 4 * self.gan_dim * 16])(x)

        output = tf.keras.layers.Dense(1)(x)

        return tf.keras.Model(inputs=[input], outputs=[output])

    def build_generator_model(self):
        input = tf.keras.Input((self.latent_dim,), dtype='float32')

        x = tf.keras.layers.Dense(4 * 4 * self.gan_dim * 16)(input)
        x = tf.keras.layers.Reshape([16, self.gan_dim * 16])(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = self.__conv2d_transpose(x, self.gan_dim * 8)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = self.__conv2d_transpose(x, self.gan_dim * 4)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = self.__conv2d_transpose(x, self.gan_dim * 2)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = self.__conv2d_transpose(x, self.gan_dim)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = self.__conv2d_transpose(x, 1)
        output = tf.nn.tanh(x)

        return tf.keras.Model(inputs=[input], outputs=[output])
