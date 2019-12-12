import os
import sys
root =  '../../../..'
sys.path.append(root)
import numpy as np
import tensorflow as tf

from src.helpers import audio
from src.models.verifier.tf.vggvox.model import VggVox

# %%

use_keras = True

# %% Load playback IRs

microphone = audio.read(os.path.join(root, 'data/vs_noise_data/microphone/microphone_01.wav'))
room = audio.read(os.path.join(root, 'data/vs_noise_data/room/room_01.wav'))
speaker = audio.read(os.path.join(root, 'data/vs_noise_data/speaker/speaker_01.wav'))

print('mic      : {}'.format(microphone.shape))
print('room     : {}'.format(room.shape))
print('speaker  : {}'.format(speaker.shape))

# %% Setup the speaker model

if use_keras:  # VERSION I - Keras model
    speaker_embedding = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, 3),
        tf.keras.layers.AveragePooling2D(2),
        tf.keras.layers.Conv2D(32, 3),
        tf.keras.layers.AveragePooling2D(2),
        tf.keras.layers.Conv2D(64, 3),
        tf.keras.layers.AveragePooling2D(2),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(32)
        ], name='vggvox_keras')

else:  # VERSION II - Layers
    def speaker_embedding(x, training=False, scope='vggvox_layers'):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            n = tf.layers.conv2d(x, 16, 3)
            n = tf.layers.average_pooling2d(n, 2, 1)
            n = tf.layers.conv2d(n, 32, 3)
            n = tf.layers.average_pooling2d(n, 2, 1)
            n = tf.layers.conv2d(n, 64, 3)
            n = tf.layers.average_pooling2d(n, 3, 1)
            n = tf.reduce_mean(n, axis=[1, 2])
            n = tf.layers.dense(n, 32)
            return n
    
# %%

graph = tf.Graph()
sess = tf.Session(graph=graph)

# GAN
with graph.as_default():
    gan_root = os.path.abspath(os.path.join(root, 'data/pt_models/wavegan/tf/female/v0/'))
    saver = tf.train.import_meta_graph(os.path.join(gan_root, 'infer.meta'))
    saver.restore(sess, os.path.join(gan_root, 'model.ckpt'))

# Whole flow
with graph.as_default():
    z = graph.get_tensor_by_name('z:0')
    x = graph.get_tensor_by_name('G_z:0')
    X, p = audio.play_n_rec(x, return_placeholders=True)
    S = audio.get_tf_spectrum(X)
    e = speaker_embedding(S)

print('latent     z: {}'.format(z.shape))
print('audio      x: {}'.format(x.shape))
print('p&rec      x: {}'.format(X.shape))
print('spect      s: {}'.format(S.shape))
print('embedding  e: {}'.format(e.shape))

latent = np.random.uniform(size=(1, 100))

with graph.as_default():
    sess.run(tf.global_variables_initializer())

# %% Test generation and embedding

with graph.as_default():

    out = sess.run(e, feed_dict={
        z: latent,
        p['microphone']: microphone,
        p['room']: room,
        p['speaker']: speaker,
        })

print('output    : {} -> {:.30s}...'.format(out.shape, str(out.round(3)).replace('\n', ' ') ))
    
# %% Define master voice gradients

with graph.as_default():
    h = tf.compat.v1.placeholder(tf.float32, (None, 512, 410, 1))
    e2 = speaker_embedding(h)
    cos_distance = tf.keras.losses.CosineSimilarity(axis=1)
    sim = cos_distance(e, e2)
    
    g = tf.gradients(sim, z)[0]

print('gradients  g: {}'.format(g.shape))

#%% Test gradients

tgt_spec = np.random.uniform(size=(1, 512, 410, 1)).astype(np.float32)

with graph.as_default():

    out_g = sess.run(g, feed_dict={
        z: latent,
        h: tgt_spec,
        p['microphone']: microphone,
        p['room']: room,
        p['speaker']: speaker,
        })

print('out grads   : {} -> {:.30s}...'.format(out_g.shape, str(out_g.round(3)).replace('\n', ' ') ))

# %% Print speaker verification and trainable variables

print('Speaker embedding: {}'.format(speaker_embedding))
print('Trainable variables:')
with graph.as_default():
    for v in tf.trainable_variables():
        print(' ', v.name)
    
