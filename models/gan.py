#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import json
import os

from helpers import plotting

def l2_normalize(x, eps=1e-12):
  '''
  Scale input by the inverse of it's euclidean norm
  '''
  return x / tf.linalg.norm(x + eps)


def pyramid(x, n=4):
    if n == 1:
        return [x]
    shape = np.array(x.shape[1:3])
    return [tf.image.resize(x, shape/(2**(n-i-1))) for i in range(n)]


class Spectral_Norm(tf.keras.constraints.Constraint):
    '''
    Spectral normalization of layer weights.

    CAUTION This code has not been tested!

    Uses power iteration method to calculate a fast approximation of the spectral norm (Golub & Van der Vorst)
    The weights are then scaled by the inverse of the spectral norm

    References:
    [1] Spectral Normalization for Generative Adversarial Networks, https://arxiv.org/abs/1802.05957
    [2] Derek Wilcox, https://colab.research.google.com/drive/1f2Ejlm3UmsthqQni9vFkGmTTcS_ndqXR
    '''
    def __init__(self, power_iters=1):
        self.n_iters = power_iters

    def __call__(self, w):
      flattened_w = tf.reshape(w, [w.shape[0], -1])
      u = tf.random.normal([flattened_w.shape[0]])
      v = tf.random.normal([flattened_w.shape[1]])
      for i in range(self.n_iters):
        v = tf.linalg.matvec(tf.transpose(flattened_w), u)
        v = l2_normalize(v)
        u = tf.linalg.matvec(flattened_w, v)
        u = l2_normalize(u)
      sigma = tf.tensordot(u, tf.linalg.matvec(flattened_w, v), axes=1)
      return w / sigma

    def get_config(self):
        return {'n_iters': self.n_iters}


class GAN(object):
    
    
    def __init__(self, dataset, version=None, gradient_penalty=True, clip_discriminator=False, latent_dist='normal'):
        self.dataset = dataset
        self.z_dim = 128
        self.latent_dist = latent_dist
        self.root_dir = './data/models/gan/'        
        self.gradient_penalty = gradient_penalty
        self.clip_discriminator = clip_discriminator
        self.performance = {'gen': [], 'disc': [], 'accuracy': []}
        self.prediction_history = {'real': [], 'fake': []}
        
        # Set specific version if explicit, otherwise find the last and increment
        if version is not None:
            self.version = int(version)
        else:            
            candidates = os.listdir(self.dirname(True))
            candidates = [int(c[1:]) for c in candidates]
            if len(candidates) > 0:            
                self.version = max(candidates) + 1
            else:
                self.version = 0
                
         
    @property
    def _args(self):
        return ('gender', 'version', 'gradient_penalty', 'clip_discriminator')


    def args(self):
        return {k: getattr(self, k) for k in self._args}


    def dirname(self, make=False):
        """
        Generate dirname for storing / loading the model
        
        Examples
        --------
        data/models/gan/ms-gan/digits/*
        data/models/gan/ms-gan/voxceleb-male/*
        """
        dirname = os.path.join(self.root_dir, self.model_code, self.gender)
        if hasattr(self, 'version') and self.version is not None:
            dirname = os.path.join(dirname, 'v' + str('{:03d}'.format(self.version)))
        if make and not os.path.isdir(dirname):
            os.makedirs(dirname)
        return dirname


    @property
    def model_code(self):
        raise NotImplementedError()


    def save(self, stats=True, save_full_models=False):
        """
        """
        dirname = self.dirname(True)
        if save_full_models:
            self.generator.save(os.path.join(dirname, 'generator.h5'))        
            self.discriminator.save(os.path.join(dirname, 'discriminator.h5'))
        else:
            self.generator.save_weights(os.path.join(dirname, 'generator.h5'))        
            self.discriminator.save_weights(os.path.join(dirname, 'discriminator.h5'))
        if stats:
            with open(os.path.join(dirname, 'stats.json'), 'w') as f:
                json.dump({
                    'performance': {k: [float(x) for x in v] for k, v in self.performance.items()}, 
                    'history': {k: [float(x) for x in v] for k, v in self.prediction_history.items()},
                    'class': self.__class__.__name__,
                    'args' : self.args()
                    }, f, indent=4)


    def save_graphs(self):
        gf = os.path.join(self.dirname(True), 'generator.png')
        df = os.path.join(self.dirname(True), 'discriminator.png')
        tf.keras.utils.plot_model(self.generator, show_shapes=True, show_layer_names=False, to_file=gf)
        tf.keras.utils.plot_model(self.discriminator, show_shapes=True, show_layer_names=False, to_file=df)


    def load(self, replace_models=False):
        dirname =  self.dirname()
        
        try:
            if replace_models:
                self.generator = tf.keras.models.load_model(os.path.join(dirname, 'generator.h5'))
                self.generator.compile()
            else:
                self.generator.load_weights(os.path.join(dirname, 'generator.h5'))
        except Exception as e:
            print('ERROR Error loading generator: ' + str(e))
        
        try:
            if replace_models:
                self.discriminator = tf.keras.models.load_model(os.path.join(dirname, 'discriminator.h5'))        
                self.discriminator.compile()
            else:
                self.discriminator.load_weights(os.path.join(dirname, 'discriminator.h5'))
        except Exception as e:
            print(f'ERROR Error loading discriminator: {e}')
    
    def sample_z(self, n=1):
        if self.latent_dist == 'normal':
            return tf.random.normal([n, self.z_dim])
        elif self.latent_dist == 'uniform':
            return tf.random.uniform([n, self.z_dim], maxval=1)
        else:
            raise ValueError(f'Unsupported latent distribution: {self.latent_dist}')

    def sample(self, n=1, strip_scales=True):
        samples = self.generator(self.sample_z(n), training=False)
        if strip_scales and isinstance(samples, list):
            return samples[-1]
        else:
            return samples        


    def discriminator_loss(self, x, G_z, D_x, D_G_z):
        """
        Discriminator loss for the Wasserstein GAN (+ gradient penalty)
        
        References:
        [1] https://papers.nips.cc/paper/7159-improved-training-of-wasserstein-gans.pdf        
        """
        
        # The original GAN loss
        disc_loss =  tf.math.reduce_mean(D_G_z) - tf.math.reduce_mean(D_x)
        
        if not self.gradient_penalty:
            return disc_loss
        
        # If the output is not multi-scale, create a dummy list with a single scale
        if not isinstance(G_z, list):
            G_z = [G_z]
        
        # Generate a corresponding image pyramid
        X = pyramid(x, len(G_z))
            
        interpolates = []
        for x, g_z in zip(X, G_z):
            assert x.shape == g_z.shape
            # Sample intermediate points between x and G_z where gradient norm penalty will be enforced 
            interpolated_shape = [len(x), 1, 1, 1]
            # interpolates = x + tf.random.uniform(shape=interpolated_shape, minval=0., maxval=1.) * (g_z - x)
            interpolates.append(x + tf.random.uniform(shape=interpolated_shape, minval=0., maxval=1.) * (g_z - x))
    
        with tf.GradientTape() as disc_tape:
            disc_tape.watch(interpolates)
            disc_interp = self.discriminator(interpolates)
        
        gradients = disc_tape.gradient(disc_interp, [interpolates])[0]
        for grads in gradients:
            g_norm = tf.math.sqrt(1e-9 + tf.math.reduce_sum(tf.math.square(grads), axis=(1, 2, 3)))        
            # Add gradient penalty to the original loss
            disc_loss += (10 / len(gradients)) * tf.math.reduce_mean((g_norm - 1.) ** 2.)
        
        return disc_loss


    def generator_loss(self, D_G_z):
        """
        Method to compute the generator loss
        :param D_G_z:     Fake audio samples
        :return:          Generator loss
        """
        return -tf.math.reduce_mean(D_G_z)


    def train_step(self, x, gsteps=1, dsteps=1):
        """
        Method to perform one training step for this gan
        :param x:           Current batch data
        :return:            (generator loss, discriminator loss)
        """
    
        z = self.sample_z(len(x))
        
        for ds in range(dsteps):
            with tf.GradientTape() as tape:
                G_z = self.generator(z, training=True)
                # D_x = discriminator(x, training=True)
                D_x = self.discriminator(pyramid(x, self.n_scales), training=True)
                D_G_z = self.discriminator(G_z, training=True)
    
                disc_loss = self.discriminator_loss(x, G_z, D_x, D_G_z)
    
            grad_dis = tape.gradient(disc_loss, self.discriminator.trainable_variables)
            self.discriminator_optimizer.apply_gradients(zip(grad_dis, self.discriminator.trainable_variables))
            
            # Clip weights
            if self.clip_discriminator:
                for l in self.discriminator.layers:
                    weights = l.get_weights()
                    weights_ = []
                    for w in weights:
                        weights_.append(tf.clip_by_value(w, -.01, .01))
                    l.set_weights(weights_)
            
        for ga in range(gsteps):
            with tf.GradientTape() as tape:
                G_z = self.generator(z, training=True)
                # D_x = discriminator(x, training=True)
                D_x = self.discriminator(pyramid(x, self.n_scales), training=True)
                D_G_z = self.discriminator(G_z, training=True)
    
                gen_loss = self.generator_loss(D_G_z)
                
            grad_gen = tape.gradient(gen_loss, self.generator.trainable_variables)   
            self.generator_optimizer.apply_gradients(zip(grad_gen, self.generator.trainable_variables))
            
        return gen_loss, disc_loss


    def train(self, train_data, epochs=500, preview_interval=10, gsteps=1, dsteps=1, lr_g=1e-4, lr_d=2.5e-4):
        
        self.discriminator_optimizer = tf.keras.optimizers.Adam(lr_d)
        self.generator_optimizer = tf.keras.optimizers.Adam(lr_g)
        
        print('', end='', flush=True)
        pbar_desc = str(self)
        
        with tqdm(total=epochs+1, desc=pbar_desc, ncols=200) as pbar:
            
            for epoch in range(epochs+1):
                # tf.keras.backend.set_learning_phase(1)
        
                gen_losses = []
                disc_losses = []
                d_real = []
                d_fake = []
                accuracy = []
                
                for step, batch_data in enumerate(train_data):
                    # Training step
                    gen_loss, disc_loss = self.train_step(batch_data, gsteps, dsteps)
                    
                    # Log current losses
                    gen_losses.append(gen_loss.numpy())
                    disc_losses.append(disc_loss.numpy())
                    
#                    z = tf.random.normal([len(batch_data), self.z_dim])
                    z = self.sample_z(len(batch_data))
                    G_z = self.generator(z) 
        
                    # Log discriminator output for real and fake samples
                    d_r = self.discriminator(pyramid(batch_data, self.n_scales), training=False)
                    d_f = self.discriminator(G_z, training=False)
                    
                    d_r = tf.keras.activations.sigmoid(d_r)
                    d_f = tf.keras.activations.sigmoid(d_f)
                    
                    d_real.append(np.mean(d_r))
                    d_fake.append(np.mean(d_f))
                    acc = 0.5 * np.mean(d_r >= 0.5) + 0.5 * np.mean(d_f < 0.5)
                    accuracy.append(acc)
                                        
                    # Update progress bar
                    pbar.set_postfix(gen=np.mean(gen_losses), disc=np.mean(disc_losses), acc=np.mean(accuracy))
                    
                pbar.update(1)
                    
                self.performance['gen'].append(np.mean(gen_losses))
                self.performance['disc'].append(np.mean(disc_losses))
                self.performance['accuracy'].append(np.mean(accuracy))
                self.prediction_history['real'].append(np.mean(d_real))
                self.prediction_history['fake'].append(np.mean(d_fake))
                
                if epoch % preview_interval == 0:
                    self.preview(8, save=True, epoch=epoch)
                    self.show_progress(True)
                    self.save(True, False)
            

    def preview(self, n=9, save=False, epoch=0):
        # tf.keras.backend.set_learning_phase(0)
        samples = self.generator(self.sample_z(n), training=False)
        
        if isinstance(samples, list):
            images = []
            for G_z in samples:
                for g_z in G_z:
                    images.append(g_z.numpy())
            
            fig = plotting.imsc(images, figwidth=4 * len(samples[0]), ncols=len(samples[0]))
                            
        else:
            samples = samples.numpy()
            if samples.ndim == 4:
                fig = plotting.imsc(samples, figwidth=4 * len(samples), ncols=len(samples))
            else:
                fig = plotting.waveforms(samples, spectrums=True)
            
        if save:
            filename = os.path.join(self.dirname(make=True), str('preview_{:04d}.jpg'.format(epoch)))
            plt.savefig(filename, bbox_inches='tight', quality=80)
            plt.close()
            return filename
        
        else:
            return fig


    def show_progress(self, save=False):
        fig, axes = plt.subplots(1, 3, figsize=(20, 3))
        axes[0].plot(self.performance['gen'])
        axes[0].plot(self.performance['disc'])
        axes[0].legend(['g', 'd'])
        axes[0].set_title('losses')
    
        axes[1].plot(self.performance['accuracy'])
        axes[1].set_title('accuracy')
    
        axes[2].plot(self.prediction_history['real'])
        axes[2].plot(self.prediction_history['fake'])
        axes[2].legend(['real', 'fake'])
        axes[2].set_title('discriminator on')
        
        if save:
            filename = os.path.join(self.dirname(make=True), 'progress.png')
            fig.savefig(filename, bbox_inches='tight')
            plt.close()
            return filename
        else:
            return fig


    def summarize_models(self):
        if isinstance(self.generator.outputs, list):
            print('> generator [' + str(self.generator.count_params()) + ']: Input ' + str(self.generator.input.shape[1:]) + ' -> ' + str(len(self.generator.outputs)) + ' Outputs ', end = '')
            for o in self.generator.outputs:
                print(' ' + str(o.shape[1:]), end = ' ')
        else:
            print('> generator: ' + str(self.generator.input.shape[1:]) + ' -> ' + str(self.generator.output.shape[1:]))

        if isinstance(self.discriminator.input, list):
            print('\n> discriminator [' + str(self.discriminator.count_params()) + ']: ' + str(len(self.discriminator.input)) + ' Inputs ', end = '')
            for o in self.discriminator.input:
                print(' ' + str(o.shape[1:]), end = '')
                print(' -> ' + str(self.discriminator.outputs[0].shape[1:]) + ':', end=' ')
        else:
            print('\n> discriminator [' + str(self.discriminator.count_params()) + ']: ' + str(self.discriminator.input.shape[1:]) + ' -> ' + str(self.discriminator.output.shape[1:]))


    def get_generator(self):
        return self.generator


class DCGAN(GAN):
    
    def __init__(self, dataset, version=None, z_dim=128, patch=32, width_ratio=1, kernel_size=5, gan_dim=16, bn=True, sn=False, gradient_penalty=True, clip_discriminator=False, latent_dist='normal'):
        super().__init__(dataset, version, gradient_penalty, clip_discriminator, latent_dist)
        self.n_scales = 1
        
        self.z_dim = z_dim
        self.patch = patch
        self.width_ratio = width_ratio
        self.kernel_size = kernel_size
        self.gan_dim = gan_dim
        self.bn = bn
        self.sn = sn
        self.d_layers = 4
        self.n_layers = int(np.log2(patch) - 2)

        # Discriminator ---------------------------------------------------------------------------
        input = tf.keras.Input((patch, int(patch * width_ratio), 1))
        constraint = Spectral_Norm() if sn else None
        
        x = input
        m = 2
        for n in range(self.d_layers):
            x = tf.keras.layers.Conv2D(gan_dim * m, kernel_size, 2, padding='SAME', use_bias=not bn, kernel_constraint=constraint)(x)
            x = tf.keras.layers.LeakyReLU()(x)
            m = m * 2
    
        x = tf.keras.layers.Flatten()(x)
    
        output = tf.keras.layers.Dense(1, kernel_constraint=constraint)(x) # activation='sigmoid'
    
        self.discriminator = tf.keras.Model(inputs=[input], outputs=[output])
    
        # Generator -------------------------------------------------------------------------------
        m = 8
        input = tf.keras.Input((z_dim,))        
    
        x = tf.keras.layers.Dense(4 * 4 * gan_dim * 16)(input)
        x = tf.keras.layers.Reshape([4, 4, gan_dim * 16])(x)
        x = tf.keras.layers.BatchNormalization()(x) if bn else x
        x = tf.keras.layers.LeakyReLU()(x)
        
        for n in range(self.n_layers):
            x = tf.keras.layers.Conv2DTranspose(gan_dim * m, kernel_size, strides=(2, 2), use_bias=not bn, padding='SAME')(x)
            x = tf.keras.layers.BatchNormalization()(x) if bn else x
            x = tf.keras.layers.LeakyReLU()(x)
            m = m // 2
            
        x = tf.keras.layers.Conv2D(1, kernel_size, strides=(1, 1), use_bias=False, padding='SAME')(x)
        # output = tf.nn.tanh(x)
    
        self.generator = tf.keras.Model(inputs=input, outputs=x)


    @property
    def _args(self):
        return super()._args + ('z_dim', 'patch', 'width_ratio', 'kernel_size', 'gan_dim', 'bn', 'sn')


    @property
    def model_code(self):
        return 'dc-gan'


class MultiscaleGAN(GAN):

    def __init__(self, dataset, version=None, z_dim=128, patch=256, width_ratio=1, kernel_size=5, bn=True, drop=0, sn=False, min_output=8, up='conv', gp=True, cd=False, latent_dist='normal'):
        """
        A Multiscale GAN with direct connection of (multi-scale) generator outputs to corresponding feature maps in the discriminator.

        # References:
        [1] MSG-GAN: Multi-Scale Gradients for Generative Adversarial Networks, https://arxiv.org/abs/1903.06048
        """
        super().__init__(dataset, version, gp, cd, latent_dist)
        
        if up not in {'conv', 'interp'}:
            raise ValueError(f'Invalid upsampling: {up}!')
            
        if width_ratio not in {0.5, 1, 2, 3, 4, 5, 6, 7, 8}:
            raise ValueError(f'Invalid width ratio: {width_ratio}!')
        
        # Save hyperparameters for future reference
        self.g_layers = int(np.log2(min(patch, patch * width_ratio)) - 2)
        self.d_layers = self.g_layers
        self.z_dim = z_dim
        self.patch = patch
        self.width_ratio = width_ratio
        self.min_output = min_output
        self.bn = bn
        self.drop = drop
        self.sn = sn
        self.up = up
        self.kernel_size = kernel_size
    
        # Generator ----------------------------------------------------------------------------------        
        n_conv = 2
        m = 4.0
        interpolate = up == 'interp'
        
        z = tf.keras.Input((z_dim,))
        
        x = tf.keras.layers.Dense(4 * int(4 * width_ratio) * int(z_dim * m))(z)
        x = tf.keras.layers.BatchNormalization()(x) if bn else x
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Reshape([4, int(4 * width_ratio), int(z_dim * m)])(x)
        
        outputs = []
        
        for i in range(self.g_layers):
            
            m = m / 2
            
            if interpolate:
                x = tf.image.resize(x, [2 * x.shape[1], 2 * x.shape[2]])
                x = tf.keras.layers.Conv2D(int(z_dim * m), self.kernel_size, use_bias=not bn, padding='SAME')(x)
            else:
                x = tf.keras.layers.Conv2DTranspose(int(z_dim * m), self.kernel_size, strides=(2, 2), use_bias=not bn, padding='SAME')(x)
    
            x = tf.keras.layers.BatchNormalization()(x) if bn else x
            x = tf.keras.layers.LeakyReLU()(x)
            
            if min_output is None or x.shape[1] >= min_output:
                o = tf.keras.layers.Conv2D(1, 1, use_bias=True, padding='SAME')(x)
                outputs.append(o)

        self.generator = tf.keras.Model(inputs=z, outputs=outputs)
        self.n_scales = len(outputs)
    
        # Discriminator ------------------------------------------------------------------------------            
        d_dim = 16
        m = 1
        n_conv = 1
        constraint = Spectral_Norm() if sn else None
        
        inputs = [tf.keras.Input(o.shape[1:]) for o in outputs]
        
        # Take the highest resolution input
        x = inputs[-1]
        x = tf.keras.layers.Conv2D(d_dim * m, self.kernel_size, use_bias=not bn, padding='SAME')(x)
        x = tf.keras.layers.Dropout(drop)(x) if drop > 0 else x
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.AveragePooling2D()(x)        
        
        for i in range(self.d_layers):
            
            if min_output is None or x.shape[1] >= min_output:
                x = tf.keras.layers.Concatenate()([inputs[self.g_layers - i - 2], x])
            
            for _ in range(n_conv):
                x = tf.keras.layers.Conv2D(d_dim * m, self.kernel_size, use_bias=not bn, padding='SAME', kernel_constraint=constraint)(x)
                x = tf.keras.layers.Dropout(drop)(x) if drop > 0 else x
                x = tf.keras.layers.LeakyReLU()(x)
                    
            x = tf.keras.layers.AveragePooling2D()(x)
    
            m = m * 2
        
        x = tf.keras.layers.Flatten()(x)
        d = tf.keras.layers.Dense(1, kernel_constraint=constraint)(x)
        
        self.discriminator = tf.keras.Model(inputs=inputs, outputs=[d])

    @property
    def _args(self):
        return super()._args + ('z_dim', 'patch', 'width_ratio', 'kernel_size', 'bn', 'sn', 'drop', 'min_output', 'up')
        
    @property
    def model_code(self):
        return 'ms-gan'

    def __str__(self):
        args = [
            f'{self.dataset}', 
            f'z={self.z_dim}', 
            f'o={self._scales}', 
            f'up={self.up}', 
            f'bn={self.bn!s:.1}', 
            f'sn={self.sn!s:.1}', 
            f'gp={self.gradient_penalty!s:.1}',
            f'clip={self.clip_discriminator!s:.1}'
        ]
        return f'MS-GAN[{",".join(args)}]'
        
