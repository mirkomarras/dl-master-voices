#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import json
import os

from helpers import plotting


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(-.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis=raxis)

class Autoencoder(tf.keras.Model):
    
    def __init__(self, dataset, version=None, z_dim=128, patch_size=256):
        super(Autoencoder, self).__init__()
        self.dataset = dataset
        self.z_dim = z_dim
        self.patch_size = patch_size

        # self.latent_dist = latent_dist
        self.build_models()
        self.reset_metrics()
        
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
    def root_dir(self):
        return './data/models/ae/'

    def reset_metrics(self):
        self.performance = {'loss': [], 'mse': []}

    def build_models(self):
        # n_layers = np.log2(128) - np.log2(32)
        z_res = self.patch_size // 8

        self.encoder = tf.keras.Sequential([
            tf.keras.Input((self.patch_size, self.patch_size, 1)),
            tf.keras.layers.Conv2D(32, 3, activation=tf.nn.leaky_relu, strides=(2,2), padding='same'),
            tf.keras.layers.Conv2D(64, 3, activation=tf.nn.leaky_relu, strides=(2,2), padding='same'),
            tf.keras.layers.Conv2D(128, 3, activation=tf.nn.leaky_relu, strides=(2,2), padding='same'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(self.z_dim)
        ])
        
        self.decoder = tf.keras.Sequential([
            tf.keras.Input((self.z_dim,)),
            tf.keras.layers.Dense(z_res * z_res * 64, activation=tf.nn.leaky_relu),
            tf.keras.layers.Reshape(target_shape=(z_res, z_res, 64)),
            tf.keras.layers.Conv2DTranspose(128, 3, strides=2, activation=tf.nn.leaky_relu, padding='same'),
            tf.keras.layers.Conv2DTranspose(64, 3, strides=2, activation=tf.nn.leaky_relu, padding='same'),
            tf.keras.layers.Conv2DTranspose(32, 3, strides=2, activation=tf.nn.leaky_relu, padding='same'),
            tf.keras.layers.Conv2DTranspose(1, 1, strides=1)
        ])

    @property
    def _args(self):
        return ('version', 'z_dim', 'patch_size')

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
        dirname = os.path.join(self.root_dir, self.model_code, self.dataset)
        if hasattr(self, 'version') and self.version is not None:
            dirname = os.path.join(dirname, 'v' + str('{:03d}'.format(self.version)))
        if make and not os.path.isdir(dirname):
            os.makedirs(dirname)
        return dirname


    @property
    def model_code(self):
        return f'ae_{self.z_dim}'


    def save(self, stats=True, save_full_models=False):
        """
        """
        dirname = self.dirname(True)
        if save_full_models:
            self.encoder.save(os.path.join(dirname, 'encoder.h5'))        
            self.decoder.save(os.path.join(dirname, 'decoder.h5'))
        else:
            self.encoder.save_weights(os.path.join(dirname, 'encoder.h5'))        
            self.decoder.save_weights(os.path.join(dirname, 'decoder.h5'))
        if stats:
            with open(os.path.join(dirname, 'stats.json'), 'w') as f:
                json.dump({
                    'performance': {k: [float(x) for x in v] for k, v in self.performance.items()}, 
                    'class': self.__class__.__name__,
                    'args' : self.args()
                    }, f, indent=4)

    def save_graphs(self):
        gf = os.path.join(self.dirname(True), 'encoder.png')
        df = os.path.join(self.dirname(True), 'decoder.png')
        tf.keras.utils.plot_model(self.encoder, show_shapes=True, show_layer_names=False, to_file=gf)
        tf.keras.utils.plot_model(self.decoder, show_shapes=True, show_layer_names=False, to_file=df)

    def preview_latent(self, x, z_std, counter=0):
        z = self.encoder(x)

        z_dim = min(self.z_dim, 10)
        z_rep = 11
        z_std = z_std if z_std is not None else np.std(z)
        X_vis = []

        for z_id in range(z_dim):
            z_delta = np.zeros((z_rep, self.z_dim))
            z_delta[:, z_id] = np.linspace(-3, 3, z_rep) * z_std[z_id]
            
            X = self.decoder(tf.repeat(z, z_rep, axis=0) + z_delta)
            X_vis.append(X)
        
        from helpers import plotting

        fig = plotting.imsc(tf.concat(X_vis, axis=0).numpy(), '', ncols=z_rep)
        fig.savefig(os.path.join(self.dirname(True), f'latent_{counter:04d}.jpg'), bbox_inches='tight', quality=80)

    def load(self, replace_models=False):
        dirname =  self.dirname()
        
        try:
            if replace_models:
                self.encoder = tf.keras.models.load_model(os.path.join(dirname, 'encoder.h5'))
                self.encoder.compile()
            else:
                self.encoder.load_weights(os.path.join(dirname, 'encoder.h5'))
        except Exception as e:
            print('ERROR Error loading generator: ' + str(e))
        
        try:
            if replace_models:
                self.decoder = tf.keras.models.load_model(os.path.join(dirname, 'decoder.h5'))
                self.decoder.compile()
            else:
                self.decoder.load_weights(os.path.join(dirname, 'decoder.h5'))
        except Exception as e:
            print(f'ERROR Error loading discriminator: {e}')
    
    def sample_z(self, n=1):
        return tf.random.normal([n, self.z_dim])

    def sample(self, n=1, strip_scales=True):
        samples = self.generator(self.sample_z(n), training=False)
        return samples        

    def encode(self, x):
        return self.encoder(x)
        
    def decode(self, z):
        return self.decoder(z)

    def codec(self, x):
        return self.decode(self.encode(x))
        
    def loss(self, x):
        x = tf.cast(x, tf.float32)
        z = self.encode(x)
        X = self.decode(z)
        return tf.reduce_mean(tf.pow(x - X, 2))

    @tf.function
    def training_step(self, x, opt):
        with tf.GradientTape() as tape:
            loss = self.loss(x)
        grads = tape.gradient(loss, self.trainable_variables)
        opt.apply_gradients(zip(grads, self.trainable_variables))
        return loss

    def train(self, ds, epochs, preview_interval=10):
        
        # self.reset_metrics()
            
        opt = tf.keras.optimizers.Adam()
        loss_m = tf.keras.metrics.Mean()
        
        with tqdm(total=epochs, desc='AE', ncols=140) as pbar:
            for epoch in range(1, epochs+1):
                loss_m.reset_states()
                # for example in ds:
                for step, batch_data in enumerate(ds):
                    x = batch_data
                    loss = self.training_step(x, opt)
                    loss_m.update_state(loss)

                loss_v = loss_m.result().numpy()
                pbar.set_postfix(loss=loss_v)
                self.performance['loss'].append(loss_v)
                pbar.update(1)
            
                if epoch % preview_interval == 0:
                    self.preview(x, save=True, epoch=epoch)
                    self.show_progress(True)
                    self.preview_latent(x[0:1], np.std(self.encode(x), axis=0))
                    self.save(True, False)

    def preview(self, x, save=False, epoch=0):

        if len(x) > 16:
            x = x[:16]

        # tf.keras.backend.set_learning_phase(0)
        samples = self.codec(x)
        samples = tf.concat((x, samples), axis=0)
        
        if isinstance(samples, list):
            images = []
            for G_z in samples:
                for g_z in G_z:
                    images.append(g_z.numpy())
            
            fig = plotting.imsc(images, figwidth=4 * len(samples[0]), ncols=len(samples[0]))
                            
        else:
            samples = samples.numpy()
            if samples.ndim == 4:
                fig = plotting.imsc(samples, figwidth=4 * len(samples), ncols=len(x))
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
        # fig, axes = plt.subplots(1, 1, figsize=(20, 3))
        fig = plt.figure()
        axes = [fig.gca()]
        # fig, axes = plotting.sub(1)
        axes[0].plot(self.performance['loss'])
        # axes[0].legend(['g', 'd'])
        axes[0].set_title('loss')
            
        if save:
            filename = os.path.join(self.dirname(make=True), 'progress.png')
            fig.savefig(filename, bbox_inches='tight')
            plt.close()
            return filename
        else:
            return fig


    def summarize_models(self):
        if isinstance(self.encoder.outputs, list):
            print(f'> encoder [{self.encoder.count_params():,d}]: Input ' + str(self.encoder.input.shape[1:]) + ' -> ' + str(len(self.encoder.outputs)) + ' Outputs ', end = '')
            for o in self.encoder.outputs:
                print(' ' + str(o.shape[1:]), end = ' ')
        else:
            print('> encoder: ' + str(self.encoder.input.shape[1:]) + ' -> ' + str(self.encoder.output.shape[1:]))

        if isinstance(self.decoder.input, list):
            print(f'\n> discriminator [{self.decoder.count_params():,d}]: ' + str(len(self.decoder.input)) + ' Inputs ', end = '')
            for o in self.decoder.input:
                print(' ' + str(o.shape[1:]), end = '')
                print(' -> ' + str(self.decoder.outputs[0].shape[1:]) + ':', end=' ')
        else:
            print(f'\n> decoder [{self.decoder.count_params():,d}]: ' + str(self.decoder.input.shape[1:]) + ' -> ' + str(self.decoder.output.shape[1:]))



class VariationalAutoencoder(Autoencoder):

    def __init__(self, dataset, version=None, z_dim=128, patch_size=256):
        super(VariationalAutoencoder, self).__init__(dataset, version, z_dim, patch_size)

    @property
    def root_dir(self):
        return './data/models/vae/'

    # def reset_metrics(self):
    #     self.performance = {'loss': [], 'mse'"}

    def build_models(self):
        # n_layers = np.log2(128) - np.log2(32)
        z_depth = 32
        z_res = self.patch_size // 8

        self.encoder = tf.keras.Sequential([
            tf.keras.Input((self.patch_size, self.patch_size, 1)),
            tf.keras.layers.Conv2D(z_depth // 4, 3, activation=tf.nn.leaky_relu, strides=(2,2), padding='same'),
            tf.keras.layers.Conv2D(z_depth // 2, 3, activation=tf.nn.leaky_relu, strides=(2,2), padding='same'),
            tf.keras.layers.Conv2D(z_depth, 3, activation=tf.nn.leaky_relu, strides=(2,2), padding='same'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(self.z_dim + self.z_dim)
        ])
        
        self.decoder = tf.keras.Sequential([
            tf.keras.Input((self.z_dim,)),
            tf.keras.layers.Dense(z_res * z_res * z_depth, activation=tf.nn.leaky_relu),
            tf.keras.layers.Reshape(target_shape=(z_res, z_res, z_depth)),
            tf.keras.layers.Conv2DTranspose(z_depth // 2, 3, strides=2, activation=tf.nn.leaky_relu, padding='same'),
            tf.keras.layers.Conv2DTranspose(z_depth // 4, 3, strides=2, activation=tf.nn.leaky_relu, padding='same'),
            tf.keras.layers.Conv2DTranspose(z_depth // 4, 3, strides=2, activation=tf.nn.leaky_relu, padding='same'),
            tf.keras.layers.Conv2D(1, 3, strides=1, padding='same')
        ])

    @property
    def model_code(self):
        return f'vae_{self.z_dim}'

    def preview_latent(self, x, z_std, counter=0):
        z = self.encoder(x)

        z_dim = min(self.z_dim, 10)
        z_rep = 11
        z_std = z_std if z_std is not None else np.std(z)
        X_vis = []

        for z_id in range(z_dim):
            z_delta = np.zeros((z_rep, self.z_dim))
            z_delta[:, z_id] = np.linspace(-3, 3, z_rep) * z_std[z_id]
            
            X = self.decoder(tf.repeat(z, z_rep, axis=0) + z_delta)
            X_vis.append(X)
        
        from helpers import plotting

        fig = plotting.imsc(tf.concat(X_vis, axis=0).numpy(), '', ncols=z_rep)
        fig.savefig(os.path.join(self.dirname(True), f'latent_{counter:04d}.jpg'), bbox_inches='tight', quality=80)
    
    # def sample_z(self, n=1):
    #     return tf.random.normal([n, self.z_dim])

    def sample(self, n=1, strip_scales=True):
        samples = self.decode(self.sample_z(n), training=False)
        # if strip_scales and isinstance(samples, list):
        #     return samples[-1]
        # else:
        #     return samples        

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar
        
    def decode(self, z):
        return self.decoder(z)

    def codec(self, x):
        m, lv = self.encode(x)        
        z = self.reparameterize(m, lv)        
        return self.decode(z)

    def loss(self, x):
        x = tf.cast(x, tf.float32)
        m, lv = self.encode(x)
        
        lv = tf.clip_by_value(lv, clip_value_min=-10, clip_value_max=10)
        
        # Reparametrization trick
        z = self.reparameterize(m, lv)        
        X = self.decode(z)
        
        # Reconstruction error
        log_px_z = - tf.linalg.norm(x - X)
        
        # KL divergence
        log_pz = log_normal_pdf(z, 0., 0.)
        log_qz_x = log_normal_pdf(z, m, lv)
        
        mse = tf.reduce_mean(tf.pow(x - X, 2))
        
        return mse, -tf.reduce_mean(log_px_z + log_pz - log_qz_x)

    @tf.function
    def sample(self):
        self.decode(tf.random.normal(shape=(100, self.l_dim)))
                        
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    @tf.function
    def training_step(self, x, opt):
        with tf.GradientTape() as tape:
            mse, loss = self.loss(x)
        grads = tape.gradient(loss, self.trainable_variables)
        opt.apply_gradients(zip(grads, self.trainable_variables))
        return mse, loss

    def train(self, ds, epochs, preview_interval=10):
        
        # self.reset_metrics()
            
        opt = tf.keras.optimizers.Adam()
        loss_m = tf.keras.metrics.Mean()
        mse_m = tf.keras.metrics.Mean()

        step_id = 0
        
        with tqdm(total=epochs, desc='AE', ncols=140) as pbar:
            for epoch in range(1, epochs+1):
                loss_m.reset_states()
                mse_m.reset_states()
                # for example in ds:
                for batch_data in ds:
                    with tf.profiler.experimental.Trace('train', step_num=step_id, _r=1):
                        x = batch_data
                        mse, loss = self.training_step(x, opt)
                        loss_m.update_state(loss)
                        mse_m.update_state(mse)
                        step_id += 1

                loss_v = loss_m.result().numpy()
                mse_v = mse_m.result().numpy()
                pbar.set_postfix(loss=loss_v, mse=mse_v)
                self.performance['loss'].append(loss_v)
                self.performance['mse'].append(mse_v)
                pbar.update(1)
            
                if epoch % preview_interval == 0:
                    self.preview(x, save=True, epoch=epoch)
                    self.show_progress(True)
                    # self.preview_latent(x[0:1], np.std(self.encode(x), axis=0))
                    self.save(True, False)

    def preview(self, x, save=False, epoch=0):

        if len(x) > 16:
            x = x[:16]

        # tf.keras.backend.set_learning_phase(0)
        samples = self.codec(x)
        samples = tf.concat((x, samples), axis=0)
        
        if isinstance(samples, list):
            images = []
            for G_z in samples:
                for g_z in G_z:
                    images.append(g_z.numpy())
            
            fig = plotting.imsc(images, figwidth=4 * len(samples[0]), ncols=len(samples[0]))
                            
        else:
            samples = samples.numpy()
            if samples.ndim == 4:
                fig = plotting.imsc(samples, figwidth=4 * len(samples), ncols=len(x))
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
        # fig, axes = plt.subplots(1, 1, figsize=(20, 3))
        # fig = plt.figure()
        # axes = [fig.gca()]
        fig, axes = plotting.sub(2)
        axes[0].plot(self.performance['loss'])
        # axes[0].legend(['g', 'd'])
        axes[0].set_title('loss')
        axes[1].plot(self.performance['mse'])
        # axes[0].legend(['g', 'd'])
        axes[1].set_title('mse')
            
        if save:
            filename = os.path.join(self.dirname(make=True), 'progress.png')
            fig.savefig(filename, bbox_inches='tight')
            plt.close()
            return filename
        else:
            return fig

def get_model(netg, gender=None):

    MODEL_DICT = {'ae': Autoencoder, 'vae': None}

    if '/v' in netg:
        arch, version = netg.split('/v')
        version = int(version)
    else:
        version = -1
    assert arch in MODEL_DICT, "Specified model is not supported!"

    return MODEL_DICT[arch](gender, version=version)
