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
        X = self.decode(z)
        return tf.reduce_mean(tf.pow(x - X, 2))
    
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

ae = AE(10)

print(ae.encoder(example).shape)

z = ae.encoder(example)

print(ae.decoder(z).shape)

print(ae.loss(example['image']))

# %%

ae.train(ds, 100)

# %%

z = ae.encoder(example)
X = ae.decoder(z)

from helpers import plotting

plotting.imsc(x[:16].numpy())
plotting.imsc(X[:16].numpy())


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

# plotting.imsc(x[:16].numpy())
# plotting.imsc(X.numpy(), '', ncols=z_rep)

fig = plotting.imsc(tf.concat(X_vis, axis=0).numpy(), '', ncols=z_rep)

# %% Can I sample?

X = ae.decoder(500 * tf.random.normal(shape=(10, 10), dtype=tf.float32))
plotting.imsc(X.numpy(), '', ncols=z_rep)