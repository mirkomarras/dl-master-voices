import os
import sys
root = './'
sys.path.append(root)
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
from helpers import audio

# %%

use_keras = True
tgt_length = 410
gan_model = ('male', 'female', 'neutral')[-1]
latent_space = ('uniform', 'normal')[-1]
sampling = 16000

# %% Load playback IRs

microphone = audio.read(os.path.join(root, 'data/vs_noise_data/microphone/microphone_01.wav'), as_filter=True)
room = audio.read(os.path.join(root, 'data/vs_noise_data/room/room_01.wav'), as_filter=True)
speaker = audio.read(os.path.join(root, 'data/vs_noise_data/speaker/speaker_01.wav'), as_filter=True)

print('\n## Loaded Environment:')
print('mic      : {}'.format(microphone.shape))
print('room     : {}'.format(room.shape))
print('speaker  : {}'.format(speaker.shape))

# %%

graph = tf.Graph()
sess = tf.Session(graph=graph)

# Whole flow
with graph.as_default():
    x = tf.compat.v1.placeholder(tf.float32, (None, None, 1))
    X, p = audio.play_n_rec(x, return_placeholders=True, noise_strength=0)

print('audio      x: {}'.format(x.shape))
print('p&rec      x: {}'.format(X.shape))

with graph.as_default():
    sess.run(tf.global_variables_initializer())

# %% Test data along the way

xt = audio.read(os.path.join(root, 'data/voxceleb/test/id10273/5TWpQYtboq0/00001.wav'), as_filter=False)

print('sample    x: {}'.format(xt.shape))

fd = {
    x: xt,
    p['microphone']: microphone,
    p['room']: room,
    p['speaker']: speaker,
}

x_, X_ = sess.run([x, X], feed_dict=fd)

print('\n## Data Flow in the Model:')
print('!audio       x: {} -> [{:.2f}, {:.2f}] // {:.1f} s'.format(x_.shape, x_.min(), x_.max(), x_.size / sampling))
print('!p&rec       X: {} -> [{:.2f}, {:.2f}] // {:.1f} s'.format(X_.shape, X_.min(), X_.max(), X_.size / sampling))

# %% Plot signals
fig, axes = plt.subplots(2, 1)
fig.set_size_inches((16, 8))

axes[0].plot(x_.ravel())
axes[0].set_title('speech sample')

axes[1].plot(X_.ravel())
axes[1].set_title('Playback sample')

plt.show()
