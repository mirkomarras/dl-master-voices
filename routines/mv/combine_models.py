#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import sys
import os

from helpers import audio
from models.verifier.vggvox import VggVox

# %% Load playback IRs

microphone = audio.read(os.path.join('data/vs_noise_data/microphone/microphone_01.wav'))
room = audio.read(os.path.join('data/vs_noise_data/room/room_01.wav'))
speaker = audio.read(os.path.join('data/vs_noise_data/speaker/speaker_01.wav'))

print('mic      : {}'.format(microphone.shape))
print('room     : {}'.format(room.shape))
print('speaker  : {}'.format(speaker.shape))

speaker_embedding = VggVox().net()

# %%

graph = tf.Graph()
sess = tf.compat.v1.Session(graph=graph)

# GAN
with graph.as_default():
    gan_root = os.path.abspath(os.path.join('data/pt_models/wavegan/female/v0/'))
    saver = tf.compat.v1.train.import_meta_graph(os.path.join(gan_root, 'infer.meta'))
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
    sess.run(tf.compat.v1.global_variables_initializer())

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
    for v in tf.compat.v1.trainable_variables():
        print(' ', v.name)
    
