#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import argparse
import sys
import os

import matplotlib.pyplot as plt

from models.verifier.resnet50vox import ResNet50Vox
from models.verifier.resnet34vox import ResNet34Vox
from models.verifier.xvector import XVector
from models.verifier.vggvox import VggVox
from models.gan.wavegan import WaveGAN
from models.gan.specgan import SpecGAN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main():
    parser = argparse.ArgumentParser(description='Master voice optimization')

    parser.add_argument('--net_verifier', dest='net_verifier', default='xvector', type=str, choices=['vggvox', 'xvector', 'resnet50vox', 'resnet34vox'], action='store', help='Network model architecture')
    parser.add_argument('--version_verifier', dest='version_verifier', default=0, type=int, action='store', help='Version of the model to resume')
    parser.add_argument('--sample_rate', dest='sample_rate', default=16000, type=int, action='store', help='Sample rate audio')
    parser.add_argument('--n_seconds', dest='n_seconds', default=3, type=int, action='store', help='Segment lenght in seconds')

    parser.add_argument('--net_gan', dest='net_gan', default='wavegan', type=str, choices=['wavegan', 'specgan'], action='store', help='Network model architecture')
    parser.add_argument('--version_gan', dest='version_gan', default=13, type=int, action='store', help='Version of the model to resume')
    parser.add_argument('--gender', dest='gender', default='neutral', type=str, choices=['neutral', 'male', 'female'], action='store', help='Training gender')
    parser.add_argument('--latent_dim', dest='latent_dim', default=100, type=int, action='store', help='Number of dimensions of the latent space')
    parser.add_argument('--slice_len', dest='slice_len', default=16384, type=int, choices=[16384, 32768, 65536], action='store', help='Number of dimensions of the latent space')

    args = parser.parse_args()

    print('Parameters summary')

    print('>', 'Net Verifier: {}'.format(args.net_verifier))
    print('>', 'Version Verifier: {}'.format(args.version_verifier))
    print('>', 'Sample rate: {}'.format(args.sample_rate))
    print('>', 'Max number of seconds: {}'.format(args.n_seconds))

    print('>', 'Net GAN: {}'.format(args.net_gan))
    print('>', 'Version GAN: {}'.format(args.version_gan))
    print('>', 'Gender: {}'.format(args.gender))
    print('>', 'Latent dim: {}'.format(args.latent_dim))
    print('>', 'Slice len: {}'.format(args.slice_len))

    print('Loading verifier')
    available_verifiers = {'xvector': XVector, 'vggvox': VggVox, 'resnet50vox': ResNet50Vox, 'resnet34vox': ResNet34Vox}
    selected_verifier = available_verifiers[args.net_verifier](id=args.version_verifier, n_seconds=args.n_seconds, sample_rate=args.sample_rate)
    selected_verifier.build(classes=2)
    verifier = selected_verifier.get_model()

    print('Loading generator')
    available_generators = {'wavegan': WaveGAN, 'specgan': SpecGAN}
    selected_generator = available_generators[args.net_gan](id=args.version_gan, gender=args.gender, latent_dim=args.latent_dim, slice_len=args.slice_len)
    selected_generator.build()
    generator = selected_generator.get_generator()

    print('Feeding combined model')
    latent = np.random.normal(size=(1, 100)).astype(np.float32)
    print('>', generator(latent))

    audio = np.random.normal(size=(1, 48000, 1)).astype(np.float32)
    impulse = np.zeros(3)
    print('>', verifier([audio, impulse]))

    print('Combining models')
    master_voicer = tf.keras.Model(inputs=[generator.layers[0].input], outputs=verifier.layers[0]([generator.layers[-1], tf.constant([0,0,0])]))












    print('Creating siamese model')
    input_a = Input(shape=(512,300,1,))
    input_b = Input(shape=(512,300,1,))
    embedding_a = verifier(input_a)
    embedding_b = verifier(input_b)
    cos_distance = tf.keras.losses.CosineSimilarity(axis=1)
    sim = cos_distance(embedding_a, embedding_b)

    g = tf.gradients(sim, generator.layers[0].input)[0]
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

if __name__ == '__main__':
    main()