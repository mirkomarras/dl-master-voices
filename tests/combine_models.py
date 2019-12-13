import os
import sys
root =  './'
sys.path.append(root)
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
from helpers import audio
from models.verifier.vggvox import VggVox

# %%

use_keras = True
tgt_length = 410
gan_model = ('male', 'female', 'neutral')[-1]
latent_space = ('uniform', 'normal')[0]
sampling = 16000
refeed_target = True
test_gradients = False

# %% Load playback IRs

microphone = audio.read(os.path.join(root, 'data/vs_noise_data/microphone/microphone_01.wav'))
room = audio.read(os.path.join(root, 'data/vs_noise_data/room/room_01.wav'))
speaker = audio.read(os.path.join(root, 'data/vs_noise_data/speaker/speaker_01.wav'))

print('\n## Loaded Environment:')
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
            n = tf.layers.average_pooling2d(n, 2, 2)
            n = tf.layers.conv2d(n, 32, 3)
            n = tf.layers.average_pooling2d(n, 2, 2)
            n = tf.layers.conv2d(n, 64, 3)
            n = tf.layers.average_pooling2d(n, 3, 2)
            n = tf.reduce_mean(n, axis=[1, 2])
            n = tf.layers.dense(n, 32)
            return n
    
# %%

graph = tf.Graph()
sess = tf.Session(graph=graph)

# GAN
with graph.as_default():
    gan_root = os.path.abspath(os.path.join(root, 'data/pt_models/wavegan/tf/{}/v0/'.format(gan_model)))
    saver = tf.train.import_meta_graph(os.path.join(gan_root, 'infer.meta'))
    saver.restore(sess, os.path.join(gan_root, 'model.ckpt'))

# Whole flow
with graph.as_default():
    z = graph.get_tensor_by_name('z:0')
    x = graph.get_tensor_by_name('G_z:0')
    X, p = audio.play_n_rec(x, return_placeholders=True)
    S = audio.get_tf_spectrum(X)
    e = speaker_embedding(S)

print('\n## TF Variables:')
print('latent     z: {}'.format(z.shape))
print('audio      x: {}'.format(x.shape))
print('p&rec      x: {}'.format(X.shape))
print('spect      s: {}'.format(S.shape))
print('embedding  e: {}'.format(e.shape))

if latent_space == 'normal':
    latent = np.random.normal(size=(1, 100))
elif latent_space == 'uniform':
    latent = np.random.uniform(size=(1, 100), low=-1, high=1)
else:
    print('error: unknown latent space!')
    sys.exit(1)

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
    h = tf.compat.v1.placeholder(tf.float32, (None, 257, tgt_length, 1))
    e2 = speaker_embedding(h)
    cos_distance = tf.keras.losses.CosineSimilarity(axis=1)
    sim = cos_distance(e, e2)
    
    g = tf.gradients(sim, z)[0]

print('gradients  g: {}'.format(g.shape))

# %% Test data along the way

print('\n## Target sample:')
xt = audio.read(os.path.join(root, 'data/voxceleb/test/id10273/5TWpQYtboq0/00001.wav'))
St, _, _ = audio.get_fft_spectrum(xt.ravel(), sampling)
print('tgt_speech xt: {} -> [{:.2f}, {:.2f}]'.format(xt.shape, xt.min(), xt.max()))
print('tgt_spect  xt: {} -> [{:.2f}, {:.2f}]'.format(St.shape, St.min(), St.max()))

if St.shape[-1] > tgt_length:
    print('warning: clipping target speech to {} samples!'.format(tgt_length))
    St = St[:, :tgt_length]

St = St.reshape((1, 257, -1, 1))

fd = {
        z: latent,
        h: St,
        p['microphone']: microphone,
        p['room']: room,
        p['speaker']: speaker,
}

if refeed_target:
    fd[x] = xt[:65536].reshape((1, -1, 1))

z_, x_, X_, S_, e_, et_ = sess.run([z, x, X, S, e, e2], feed_dict=fd)

print('\n## Data Flow in the Model:')
print('!latent space: {} -> {} GAN'.format(latent_space, gan_model))
print('!latent      z: {} -> [{:.2f}, {:.2f}]'.format(z_.shape, z_.min(), z_.max()))
print('!audio       x: {} -> [{:.2f}, {:.2f}] // {:.1f} s'.format(x_.shape, x_.min(), x_.max(), x_.size / sampling))
print('!p&rec       X: {} -> [{:.2f}, {:.2f}] // {:.1f} s'.format(X_.shape, X_.min(), X_.max(), X_.size / sampling))
print('!spect       S: {} -> [{:.2f}, {:.2f}]'.format(S_.shape, S_.min(), S_.max()))
print('!embedding   e: {} -> [{:.2f}, {:.2f}]'.format(e_.shape, e_.min(), e_.max()))
print('!tgt speech xt: {} -> [{:.2f}, {:.2f}] // {:.1f} s'.format(xt.shape, xt.min(), xt.max(), xt.size / sampling))
print('!tgt spect  St: {} -> [{:.2f}, {:.2f}]'.format(St.shape, St.min(), St.max()))
print('!tgt embe   et: {} -> [{:.2f}, {:.2f}]'.format(et_.shape, et_.min(), et_.max()))


# %% Plot signals
fig, axes = plt.subplots(4, 2)
fig.set_size_inches((12, 16))

axes[0,0].hist(z_.ravel())
axes[0,0].set_title('Latent distribution')
axes[0,1].plot(x_.ravel()[:sampling])
axes[0,1].set_title('GAN-generated sample [1st sec]')

axes[1,1].plot(X_.ravel()[:sampling])
axes[1,1].set_title('Playback of GAN sample [1st sec]')
axes[1,0].plot(xt.ravel()[:sampling])
axes[1,0].set_title('Target speech [1st sec]')

axes[2,1].imshow(S_.reshape(257, -1)[:, :256], aspect='auto')
axes[2,1].set_title('Spec GAN sample [{:.2f}, {:.2f}]'.format(S_.min(), S_.max()))
axes[2,0].imshow(St.reshape(257, -1)[:, :256], aspect='auto')
axes[2,0].set_title('Target spec [{:.2f}, {:.2f}]'.format(St.min(), St.max()))

axes[3,0].plot(e_.ravel(), et_.ravel(), 'o')
axes[3,0].set_xlabel('src embedding')
axes[3,0].set_ylabel('tgt embedding')

axes[3,1].hist(S_.ravel(), 30, alpha=0.5)
axes[3,1].hist(St.ravel(), 30, alpha=0.5)
axes[3,1].legend(['GAN spect', 'Target spect'])

plt.show()

# %% Test gradients

if test_gradients:

    with graph.as_default():

        out_g = sess.run(g, feed_dict=fd)

    print('\n## Gradients:')
    print('out grads   : {} -> {:.30s}...'.format(out_g.shape, str(out_g.round(3)).replace('\n', ' ') ))

    # %% Print speaker verification and trainable variables

    print('\n## SV Model Details:')
    print('Speaker embedding: {}'.format(speaker_embedding))
    print('Trainable variables:')
    with graph.as_default():
        for v in tf.trainable_variables():
            print(' ', v.name)
