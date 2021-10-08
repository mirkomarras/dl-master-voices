#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 15:57:37 2021

@author: pkorus
"""

from tqdm import tqdm
import tensorflow as tf
import tensorflow_datasets as tfds

# %%

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# %%

mnist = tfds.load('mnist')

# %%

ds = mnist['train']

ds = ds.shuffle(1024).batch(256).prefetch(tf.data.experimental.AUTOTUNE)

# ctr = 0
for example_id, example in enumerate(ds.take(1875)):
    print(example['image'].shape)

# %%

class AE(tf.keras.Model):
    
    def __init__(self, l_dim=64, patch_size=28):
        super(AE, self).__init__()
        
        l_res = patch_size // 4
        self.patch_size = patch_size
        self.latent_dim = l_dim
        
        self.encoder = tf.keras.Sequential([
            tf.keras.Input((patch_size, patch_size, 1)),
            tf.keras.layers.Conv2D(32, 3, activation=tf.nn.leaky_relu, strides=(2,2), padding='same'),
            tf.keras.layers.Conv2D(64, 3, activation=tf.nn.leaky_relu, strides=(2,2), padding='same'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(l_dim)
        ])
        
        self.decoder = tf.keras.Sequential([
            tf.keras.Input((l_dim,)),
            tf.keras.layers.Dense(l_res * l_res * 64, activation=tf.nn.leaky_relu),
            tf.keras.layers.Reshape(target_shape=(7, 7, 64)),
            tf.keras.layers.Conv2DTranspose(64, 3, strides=2, activation=tf.nn.leaky_relu, padding='same'),
            tf.keras.layers.Conv2DTranspose(32, 3, strides=2, activation=tf.nn.leaky_relu, padding='same'),
            tf.keras.layers.Conv2DTranspose(1, 1, strides=1)
        ])
        
    def encode(self, x):
        return self.encoder(x)
        
    def decode(self, z):
        return self.decoder(z)
        
    def loss(self, x):
        x = tf.cast(x, tf.float32)
        z = self.encode(x)
        X = self.decode(z + tf.random.normal(shape=z.shape))
        return tf.reduce_mean(tf.pow(x - X, 2)) + tf.reduce_mean(tf.pow(z, 2))
    
    @tf.function
    def training_step(self, x, opt):
        with tf.GradientTape() as tape:
            loss = self.loss(x)
        grads = tape.gradient(loss, self.trainable_variables)
        opt.apply_gradients(zip(grads, self.trainable_variables))
        return loss
    
    def reset_metrics(self):
        self.performance = {'loss': []}        
    
    def train(self, ds, epochs, restart=True):
        
        if restart:
            self.reset_metrics()
            
        opt = tf.keras.optimizers.Adam()
        loss_m = tf.keras.metrics.Mean()
        
        with tqdm(total=epochs, desc='AE', ncols=140) as pbar:
            for epoch in range(1, epochs+1):
                loss_m.reset_states()
                for example in ds.take(len(ds)):
                    x = example['image']
                    loss = ae.training_step(x, opt)
                    loss_m.update_state(loss)

                loss_v = loss_m.result().numpy()
                pbar.set_postfix(loss=loss_v)
                self.performance['loss'].append(loss_v)
                pbar.update(1)
    
ae = AE(2)

print(ae.encoder(example).shape)

z = ae.encoder(example)

print(ae.decoder(z).shape)

print(ae.loss(example['image']))

# %%

def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(-.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis=raxis)


class VAE(tf.keras.Model):
    
    def __init__(self, l_dim=64, patch_size=28):
        super(VAE, self).__init__()
        
        l_res = patch_size // 4
        self.patch_size = patch_size
        self.latent_dim = l_dim
        
        self.encoder = tf.keras.Sequential([
            tf.keras.Input((patch_size, patch_size, 1)),
            tf.keras.layers.Conv2D(32, 3, activation=tf.nn.leaky_relu, strides=(2,2), padding='same'),
            tf.keras.layers.Conv2D(64, 3, activation=tf.nn.leaky_relu, strides=(2,2), padding='same'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(l_dim + l_dim)
        ])
        
        self.decoder = tf.keras.Sequential([
            tf.keras.Input((l_dim,)),
            tf.keras.layers.Dense(l_res * l_res * 64, activation=tf.nn.leaky_relu),
            tf.keras.layers.Reshape(target_shape=(7, 7, 64)),
            tf.keras.layers.Conv2DTranspose(64, 3, strides=2, activation=tf.nn.leaky_relu, padding='same'),
            tf.keras.layers.Conv2DTranspose(32, 3, strides=2, activation=tf.nn.leaky_relu, padding='same'),
            tf.keras.layers.Conv2DTranspose(1, 1, strides=1)
        ])
        
    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar
        
    def decode(self, z):
        return self.decoder(z)
        
    def codec(self, x):
        m, lv = self.encode(x)        
        # Reparametrization trick
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
    
    # @tf.function
    def training_step(self, x, opt):
        with tf.GradientTape() as tape:
            mse, loss = self.loss(x)
        grads = tape.gradient(loss, self.trainable_variables)
        opt.apply_gradients(zip(grads, self.trainable_variables))
        return mse, loss
    
    def reset_metrics(self):
        self.performance = {'loss': [], 'mse': []}
    
    def train(self, ds, epochs, restart=True):
        
        if restart:
            self.reset_metrics()
            
        opt = tf.keras.optimizers.Adam()
        loss_m = tf.keras.metrics.Mean()
        mse_m = tf.keras.metrics.Mean()
        
        with tqdm(total=epochs, desc='AE', ncols=140) as pbar:
            for epoch in range(1, epochs+1):
                loss_m.reset_states()
                mse_m.reset_states()
                for example in ds.take(len(ds)):
                    x = example['image']
                    mse, loss = ae.training_step(x, opt)
                    loss_m.update_state(loss)
                    mse_m.update_state(mse)

                loss_v = loss_m.result().numpy()
                mse_v = mse_m.result().numpy()
                pbar.set_postfix(loss=loss_v, mse=mse_v)
                self.performance['loss'].append(loss_v)
                self.performance['mse'].append(mse_v)
                pbar.update(1)
    
    @tf.function
    def sample(self):
        self.decode(tf.random.normal(shape=(100, self.l_dim)))
                        
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

ae = VAE(16)

print(ae.encoder(example).shape)

z = ae.encoder(example)

print(ae.decoder(z).shape)

print(ae.loss(example['image']))

# %%

ae.train(ds, 100)

# %%

x = example['image']
# z = ae.encoder(x)
# X = ae.decoder(z)

m, lv = ae.encode(x)        
z = ae.reparameterize(m, lv)        
X = ae.decode(z)


from helpers import plotting
from matplotlib import pyplot as plt


plt.subplot(2, 1, 1)
plt.plot(ae.performance['loss'])
plt.plot(ae.performance['mse'])
plt.subplot(2, 1, 2)
plt.hist(z.numpy().ravel(), 40)

plotting.images(x[:16].numpy())

plotting.images(X[:16].numpy())


# %% Change latent vector

z = ae.encoder(example['image'][1:2])

z_id = 0
z_rep = 11
z_std = np.std(z)

X_vis = []


for z_id in range(10):
    z_delta = np.zeros((z_rep, 10))
    z_delta[:, z_id] = np.linspace(-3, 3, z_rep) * z_std
    
    X = ae.decoder(tf.repeat(z, z_rep, axis=0) + z_delta)
    X_vis.append(X)


from helpers import plotting

# plotting.images(x[:16].numpy())
# plotting.images(X.numpy(), '', ncols=z_rep)

fig = plotting.images(tf.concat(X_vis, axis=0).numpy(), '', ncols=z_rep)

# %% Can I sample?

X = ae.decoder(5 * tf.random.normal(shape=(10, 2), dtype=tf.float32))
plotting.images(X.numpy(), '', ncols=z_rep)

# %% Visualize latent space

z = ae.encoder(example['image'][1:2])

span = 10

z_id = 0
z_rep = 11
z_std = np.std(z)

X_vis = []

for z1 in np.linspace(-span, span, z_rep):
    for z2 in np.linspace(-span, span, z_rep):
        z = np.array([z1, z2]).reshape((1, -1))
        X = ae.decoder(z)
        X_vis.append(X)

# from helpers import plotting

# plotting.images(x[:16].numpy())
# plotting.images(X.numpy(), '', ncols=z_rep)

fig = plotting.images(tf.concat(X_vis, axis=0).numpy(), '', ncols=z_rep)
