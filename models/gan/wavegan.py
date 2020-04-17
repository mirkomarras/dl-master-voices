#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import os

from models.gan.model import GAN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class WaveGAN(GAN):
    """
       Class to represent WaveGAN model - Raw audio signals are returned
       Donahue, C., McAuley, J., & Puckette, M. (2018).
       Adversarial audio synthesis.
       In: arXiv preprint arXiv:1802.04208.
    """

    def __init__(self, name='wavegan', id=-1, gender='neutral', latent_dim=100, slice_len=16384):
        super().__init__(name, id, gender, latent_dim)
        self.slice_len = slice_len
        self.kernel_size = 25
        self.gan_dim = 64
        self.upsample = 'zeros'
        self.phaseshuffle = 2
        self.stride = 2 if self.slice_len == 32768 else 4
        self.is_raw = True

    def __conv1d_transpose(self, inputs, filters):
        if self.upsample == 'zeros':
            return tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=(1, self.kernel_size), strides=(1, self.stride), padding='SAME')(tf.expand_dims(inputs, axis=1))[:, 0]
        else:
            _, w, nch = inputs.get_shape().as_list()
            x = inputs
            x = tf.expand_dims(x, axis=1)
            x = tf.image.resize_nearest_neighbor(x, [1, w * self.stride])
            x = x[:, 0]
            return tf.keras.layers.Conv1D(filters, self.kernel_size, 1, padding='SAME', dtype='float32')(x)

    def __apply_phaseshuffle(self, inputs, pad_type='reflect'):
        b, x_len, nch = inputs.get_shape().as_list()
        phase = tf.random.uniform([], minval=-self.phaseshuffle, maxval=self.phaseshuffle + 1, dtype=tf.int32)
        pad_l = tf.maximum(phase, 0)
        pad_r = tf.maximum(-phase, 0)
        phase_start = pad_r
        x = tf.pad(inputs, [[0, 0], [pad_l, pad_r], [0, 0]], mode=pad_type)
        x = x[:, phase_start:phase_start+x_len]
        x.set_shape([b, x_len, nch])
        return x

    def build_discriminator_model(self):
        input = tf.keras.Input((self.slice_len,1,), dtype='float32')

        x = tf.keras.layers.Conv1D(self.gan_dim, self.kernel_size, 4, padding='SAME', dtype='float32')(input)
        x = tf.maximum(0.2 * x, x)
        x = self.__apply_phaseshuffle(x) if self.phaseshuffle > 0 else x

        x = tf.keras.layers.Conv1D(self.gan_dim * 2, self.kernel_size, 4, padding='SAME', dtype='float32')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.maximum(0.2 * x, x)
        x = self.__apply_phaseshuffle(x) if self.phaseshuffle > 0 else x

        x = tf.keras.layers.Conv1D(self.gan_dim * 4, self.kernel_size, 4, padding='SAME', dtype='float32')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.maximum(0.2 * x, x)
        x = self.__apply_phaseshuffle(x) if self.phaseshuffle > 0 else x

        x = tf.keras.layers.Conv1D(self.gan_dim * 8, self.kernel_size, 4, padding='SAME', dtype='float32')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.maximum(0.2 * x, x)
        x = self.__apply_phaseshuffle(x) if self.phaseshuffle > 0 else x

        x = tf.keras.layers.Conv1D(self.gan_dim * 16, self.kernel_size, 4, padding='SAME', dtype='float32')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.maximum(0.2 * x, x)

        if self.slice_len == 32768 or self.slice_len == 65536:
            x = tf.keras.layers.Conv1D(self.gan_dim * 32, self.kernel_size, self.stride, padding='SAME', dtype='float32')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.maximum(0.2 * x, x)

        output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        return tf.keras.Model(inputs=[input], outputs=[output])

    def build_generator_model(self):
        dim_mul = 16 if self.slice_len == 16384 else 32

        input = tf.keras.Input((self.latent_dim,), dtype='float32')

        x = tf.keras.layers.Dense(4 * 4 * self.gan_dim * dim_mul)(input)
        x = tf.keras.layers.Reshape([16, self.gan_dim * dim_mul])(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        dim_mul //= 2

        x = self.__conv1d_transpose(x, self.gan_dim * dim_mul)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        dim_mul //= 2

        x = self.__conv1d_transpose(x, self.gan_dim * dim_mul)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        dim_mul //= 2

        x = self.__conv1d_transpose(x, self.gan_dim * dim_mul)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        dim_mul //= 2

        x = self.__conv1d_transpose(x, self.gan_dim * dim_mul)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        if self.slice_len == 32768 or self.slice_len == 65536:
            x = self.__conv1d_transpose(x, self.gan_dim * dim_mul)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)

        x = self.__conv1d_transpose(x, 1)
        output = tf.nn.tanh(x)

        return tf.keras.Model(inputs=[input], outputs=[output])
