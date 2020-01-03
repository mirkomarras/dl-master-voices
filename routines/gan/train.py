#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import soundfile as sf
import numpy as np
import argparse
import sys
import os

import matplotlib.pyplot as plt

from helpers.datapipeline import data_pipeline_generator_mv, data_pipeline_mv
from helpers.dataset import get_mv_analysis_users, load_data_set, filter_by_gender
from models.verifier.resnet50vox import ResNet50Vox
from models.verifier.resnet34vox import ResNet34Vox
from models.verifier.xvector import XVector
from models.verifier.vggvox import VggVox
from models.gan.wavegan import WaveGAN
from models.gan.specgan import SpecGAN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main():
    parser = argparse.ArgumentParser(description='Master voice optimization')

    parser.add_argument('--n_iterations', dest='n_iterations', default=100, type=int, action='store', help='Number of iterations')
    parser.add_argument('--net_verifier', dest='net_verifier', default='xvector', type=str, choices=['vggvox', 'xvector', 'resnet50vox', 'resnet34vox'], action='store', help='Network model architecture')
    parser.add_argument('--version_verifier', dest='version_verifier', default=0, type=int, action='store', help='Version of the model to resume')
    parser.add_argument('--sample_rate', dest='sample_rate', default=16000, type=int, action='store', help='Sample rate audio')
    parser.add_argument('--audio_dir', dest='audio_dir', default='./data/vs_voxceleb1/dev', type=str, action='store', help='Comma-separated audio data directories')
    parser.add_argument('--gender', dest='gender', default='female', type=str, choices=['neutral', 'male', 'female'], action='store', help='Training gender')

    parser.add_argument('--audio_meta', dest='audio_meta', default='./data/ad_voxceleb12/vox12_meta_data.csv', type=str, action='store', help='CSV file with id-gender metadata')
    parser.add_argument('--mv_data_path', dest='mv_data_path', default='./data/ad_voxceleb12/vox2_mv_data.npz', type=str, action='store', help='Numpy data for master voice analysis')

    parser.add_argument('--net_gan', dest='net_gan', default='wavegan', type=str, choices=['wavegan', 'specgan'], action='store', help='Network model architecture')
    parser.add_argument('--version_gan', dest='version_gan', default=13, type=int, action='store', help='Version of the model to resume')
    parser.add_argument('--gender_gan', dest='gender_gan', default='female', type=str, choices=['neutral', 'male', 'female'], action='store', help='Training gender')
    parser.add_argument('--latent_dim', dest='latent_dim', default=100, type=int, action='store', help='Number of dimensions of the latent space')
    parser.add_argument('--slice_len', dest='slice_len', default=16384, type=int, choices=[16384, 32768, 65536], action='store', help='Number of dimensions of the latent space')

    parser.add_argument('--n_epochs', dest='n_epochs', default=1024, type=int, action='store', help='Training epochs')
    parser.add_argument('--prefetch', dest='prefetch', default=1024, type=int, action='store', help='Data pipeline prefetch size')
    parser.add_argument('--batch', dest='batch', default=32, type=int, action='store', help='Training batch size')
    parser.add_argument('--learning_rate', dest='learning_rate', default=1e-2, type=float, action='store', help='Learning rate')
    parser.add_argument('--n_seconds', dest='n_seconds', default=3, type=int, action='store', help='Segment lenght in seconds')

    args = parser.parse_args()

    print('Parameters summary')

    print('>', 'Net Verifier: {}'.format(args.net_verifier))
    print('>', 'Version Verifier: {}'.format(args.version_verifier))
    print('>', 'Sample rate: {}'.format(args.sample_rate))
    print('>', 'Audio dir: {}'.format(args.audio_dir))
    print('>', 'Master voice data path: {}'.format(args.mv_data_path))
    print('>', 'Audio meta: {}'.format(args.audio_meta))

    print('>', 'Net GAN: {}'.format(args.net_gan))
    print('>', 'Version GAN: {}'.format(args.version_gan))
    print('>', 'Gender: {}'.format(args.gender))
    print('>', 'Latent dim: {}'.format(args.latent_dim))
    print('>', 'Slice len: {}'.format(args.slice_len))

    print('>', 'Number of iterations: {}'.format(args.n_iterations))
    print('>', 'Number of epochs: {}'.format(args.n_epochs))
    print('>', 'Batch size: {}'.format(args.batch))
    print('>', 'Learning rate: {}'.format(args.learning_rate))
    print('>', 'Prefetch: {}'.format(args.prefetch))
    print('>', 'Max number of seconds: {}'.format(args.n_seconds))

    audio_dir = map(str, args.audio_dir.split(','))
    mv_user_ids = get_mv_analysis_users(args.mv_data_path, type='train')
    x_train, y_train = load_data_set(audio_dir, mv_user_ids, include=True)
    x_train, y_train = filter_by_gender(x_train, y_train, args.audio_meta, args.gender)

    print('Loading verifier')
    available_verifiers = {'xvector': XVector, 'vggvox': VggVox, 'resnet50vox': ResNet50Vox, 'resnet34vox': ResNet34Vox}
    selected_verifier = available_verifiers[args.net_verifier](id=args.version_verifier, n_seconds=args.n_seconds, sample_rate=args.sample_rate)
    selected_verifier.build(classes=2)
    verifier = selected_verifier.get_model()
    flatten_layer = tf.keras.layers.Flatten()(verifier.output)
    verifier = tf.keras.Model(verifier.inputs, flatten_layer)

    print('Loading generator')
    available_generators = {'wavegan': WaveGAN, 'specgan': SpecGAN}
    selected_generator = available_generators[args.net_gan](id=args.version_gan, gender=args.gender_gan, latent_dim=args.latent_dim, slice_len=args.slice_len)
    selected_generator.build()
    generator = selected_generator.get_generator()

    print('Setting learning phase')
    tf.keras.backend.set_learning_phase(0)
    print('> learning phase', tf.keras.backend.learning_phase())

    print('Testing model pipelining')
    latent = np.random.normal(size=(1, 100)).astype(np.float32)
    audio = generator(latent).numpy()
    impulse = np.zeros(3).astype(np.float32)
    embedding = verifier([audio, impulse])
    print('> latent', latent.shape)
    print('> audio', audio.shape)
    print('> impulse', impulse.shape)
    print('> embedding', embedding.shape)

    print('Testing generator and verifier stack')
    input_1 = generator.input
    input_2 = generator.output
    input_3 = tf.keras.layers.Input(shape=(3,))
    output = verifier([input_2, input_3])
    master_voicer = tf.keras.Model([input_1, input_3], output)

    latent = np.random.normal(size=(1, 100)).astype(np.float32)
    impulse = np.zeros(3).astype(np.float32)
    embedding = master_voicer([latent, impulse])

    print('> latent', latent.shape)
    print('> embedding', embedding.shape)

    print('Creating siamese network')
    in_a = tf.keras.layers.Input(shape=(100,))
    in_b = tf.keras.layers.Input(shape=(48000, 1,))
    in_c = tf.keras.layers.Input(shape=(3,))
    emb_a = master_voicer([in_a, in_c])
    emb_b = verifier([in_b, in_c])
    similarity = tf.keras.layers.Dot(axes=1, normalize=True)([emb_a, emb_b])
    siamese = tf.keras.Model([in_a, in_b, in_c], similarity)

    print('Testing gradient computation')
    input_1 = tf.Variable(np.random.normal(size=(1, 100)), dtype=tf.float32)
    input_2 = tf.Variable(np.random.normal(size=(1, 48000, 1)), dtype=tf.float32)
    input_3 = tf.Variable(np.zeros((1, 3)), dtype=tf.float32)
    with tf.GradientTape() as tape:
        loss = verifier([input_2, input_3])
        #print('> loss', loss.numpy())
    grads = tape.gradient(loss, input_2)
    print('> gradients', grads.shape)

    exit(1)

    print('Checking generator output')
    for index, x in enumerate(data_pipeline_generator_mv(x_train[:10], sample_rate=args.sample_rate, n_seconds=args.n_seconds)):
        print('>', index, x.shape),

    # Data pipeline output test
    print('Checking data pipeline output')
    train_data = data_pipeline_mv(x_train, sample_rate=args.sample_rate, n_seconds=args.n_seconds, batch=args.batch, prefetch=args.prefetch)

    for index, x in enumerate(train_data):
        print('>', index, x.shape)
        if index == 10:
            break

    # Create and train model
    print('Learning master voice')
    train_data = data_pipeline_mv(x_train, sample_rate=args.sample_rate, n_seconds=args.n_seconds, batch=args.batch, prefetch=args.prefetch)
    filter_gradients = lambda c, g, t1, t2: [g[i] for i in range(len(c)) if c[i] >= t1 and c[i] <= t2]

    for iter in range(args.n_iterations):
        latent_mv = np.random.normal(size=(1, 100)).astype(np.float32)
        latent_sv = np.copy(latent_mv)
        for epoch in range(args.n_epochs):
            for step, batch_data in enumerate(train_data):
                input_1 = tf.Variable(np.tile(latent_mv, (len(batch_data), 1)), dtype=tf.float32)
                input_2 = tf.Variable(batch_data, dtype=tf.float32)
                with tf.GradientTape() as tape:
                    loss = siamese([input_1, input_2])
                    tape.watch(input_1)
                grads = tape.gradient(loss, input_1)

                filtered_grads = filter_gradients(loss, grads, 0.25, 0.75)

                if len(filtered_grads) > 0:
                    perturbation = np.mean(filtered_grads, axis=0) * args.learning_rate
                    perturbation = np.clip(perturbation, 1e-5, None)
                    latent_mv += perturbation

                print('\rIter ', iter+1, 'of', args.n_iterations, 'Epoch', epoch+1, 'of', args.n_epochs, 'Step', step+1, 'of', len(x_train) // args.batch, 'loss', round(np.mean(loss), 5), end='')

            dir_mv = os.path.join('.', 'data', 'vs_mv_data', args.net_verifier + '_' + args.net_gan + '_' + args.gender_gan[0] + '-' + args.gender[0] + '_mv')
            dir_sv = os.path.join('.', 'data', 'vs_mv_data', args.net_verifier + '_' + args.net_gan + '_' + args.gender_gan[0] + '-' + args.gender[0] + '_sv')
            id_mv = str(len(os.listdir(dir_mv)))
            id_sv = str(len(os.listdir(dir_sv)))
            assert id_mv == id_sv
            if not os.path.exists(os.path.join(dir_mv, 'v' + str(id_mv))) or not os.path.exists(os.path.join(dir_sv, 'v' + str(id_sv))):
                os.makedirs(os.path.join(dir_mv, 'v' + str(id_mv)))
                os.makedirs(os.path.join(dir_sv, 'v' + str(id_sv)))
            np.save(os.path.join(dir_mv, 'v' + str(id_mv), 'sample_' + str(iter) + '.npz'), latent_mv)
            print('>', 'saved mv latent in', os.path.join(dir_mv, 'v' + str(id_mv), 'sample_' + str(iter) + '.npz'))
            sf.write(os.path.join(dir_mv, 'v' + str(id_mv), 'sample_' + str(iter) + '.wav'), generator(latent_mv).numpy(), args.sample_rate)
            print('>', 'saved mv wav in', os.path.join(dir_mv, 'v' + str(id_mv), 'sample_' + str(iter) + '.wav'))
            np.save(os.path.join(dir_sv, 'v' + str(id_sv), 'sample_' + str(iter) + '.npz'), latent_sv)
            print('>', 'saved sv latent in', os.path.join(dir_sv, 'v' + str(id_sv), 'sample_' + str(iter) + '.npz'))
            sf.write(os.path.join(dir_sv, 'v' + str(id_sv), 'sample_' + str(iter) + '.wav'), generator(latent_sv).numpy(), args.sample_rate)
            print('>', 'saved sv wav in', os.path.join(dir_sv, 'v' + str(id_sv), 'sample_' + str(iter) + '.wav'))

if __name__ == '__main__':
    main()