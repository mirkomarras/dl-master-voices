#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import argparse
import random

from src.helpers.dataset import get_mv_analysis_users, load_data_set, filter_by_gender
from src.helpers.audio import load_noise_paths, cache_noise_data, decode_audio

from src.models.verifier.tf.resnet50vox.model import ResNet50Vox
from src.models.verifier.tf.resnet34vox.model import ResNet34Vox
from src.models.verifier.tf.xvector.model import XVector
from src.models.verifier.tf.vggvox.model import VggVox

if __name__ == '__main__':

    # PARSE CLI ARGUMENTS
    parser = argparse.ArgumentParser(description='Tensorflow master voice optimization over spectrograms')

    # Network architecture arguments
    parser.add_argument('--net', dest='net', default='', type=str, choices=['vggvox', 'xvector', 'resnet50vox', 'resnet34vox'], action='store', help='Network model architecture')
    parser.add_argument('--net_version', dest='net_version', default='', type=str, action='store', help='Network version number')
    parser.add_argument('--gender', dest='gender', default='neutral', type=str, choices=['neutral', 'male', 'female'], action='store', help='Training gender')

    # Noise and audio data arguments
    parser.add_argument('--sample_rate', dest='sample_rate', default=16000, type=int, action='store', help='Sample rate audio')
    parser.add_argument('--n_seconds', dest='n_seconds', default=3, type=int, action='store', help='Segment lenght in seconds')
    parser.add_argument('--augment', dest='augment', default=0, type=int, choices=[0,1], action='store', help='Data augmentation mode')
    parser.add_argument('--noise_dir', dest='noise_dir', default='./data/vs_noise_data', type=str, action='store', help='Noise directory')
    parser.add_argument('--audio_dir', dest='audio_dir', default='./data/vs_voxceleb1/dev', type=str, action='store', help='Comma-separated audio data directories')
    parser.add_argument('--audio_meta', dest='audio_meta', default='./data/ad_voxceleb12/vox12_meta_data.csv', type=str, action='store', help='CSV file with id-gender metadata')

    # Master voice train and test meta data arguments
    parser.add_argument('--mv_data_path', dest='mv_data_path', default='./data/ad_voxceleb12/vox2_mv_data.npz', type=str, action='store', help='Numpy data for master voice analysis')
    parser.add_argument('--mv_seed_file', dest='mv_seed_file', default='', type=str, action='store', help='Path to a seed waveform')
    parser.add_argument('--n_iterations', dest='n_iterations', default=1000, type=int, action='store', help='Number of optimization stages')
    parser.add_argument('--batch', dest='batch', default=64, type=int, action='store', help='Optimization batch size')
    parser.add_argument('--min_sim', dest='min_sim', default=0.25, type=float, action='store', help='Minimum similarity')
    parser.add_argument('--max_sim', dest='max_sim', default=0.75, type=float, action='store', help='Maximum similarity')
    parser.add_argument('--learning_rate', dest='learning_rate', default=0.01, type=float, action='store', help='Optimization learning rate')

    args = parser.parse_args()

    # LOAD DATA

    # Noise data
    audio_dir = map(str, args.audio_dir.split(','))
    print('Load impulse response paths')
    noise_paths = load_noise_paths(args.noise_dir)
    print('Cache impulse response data')
    noise_cache = cache_noise_data(noise_paths, sample_rate=args.sample_rate)

    # Train data
    mv_user_ids = get_mv_analysis_users(args.mv_data_path, type='train')
    x_train, y_train = load_data_set(audio_dir, mv_user_ids, include=True)
    x_train, y_train = filter_by_gender(x_train, y_train, args.audio_meta, args.gender)

    # RESTORE MODEL
    print('Creating model')
    available_nets = {'xvector': XVector, 'vggvox': VggVox, 'resnet50vox': ResNet50Vox, 'resnet34vox': ResNet34Vox}
    model = available_nets[args.net](tf.get_default_graph(), reuse=True, id=args.net_version)
    model.build(None, None, noises=noise_paths, cache=noise_cache, augment=args.augment)

    # CREATE SIAMESE MODEL
    print('Creating siamese model')
    bottleneck_extractor = Model(bottleneck.inputs, Flatten()(bottleneck.output))
    in_a = Input(shape=(512, None, 1), name='Spectrogram01')
    in_b = Input(shape=(512, None, 1), name='Spectrogram02')
    inputs = [in_a, in_b]
    emb_a = bottleneck_extractor(in_a)
    emb_b = bottleneck_extractor(in_b)
    similarity = Dot(axes=1, normalize=True)([emb_a, emb_b])
    siamese = Model(inputs, similarity)

    model_input_layer = [siamese.layers[0].input, siamese.layers[1].input]
    model_output_layer = siamese.layers[-1].output
    cost_function = model_output_layer[0][0]
    gradient_function = K.gradients(cost_function, model_input_layer)[0]
    grab_cost_and_gradients_from_model = K.function(model_input_layer, [cost_function, gradient_function])
    filter_gradients = lambda c, g, t1, t2: [g[i] for i in range(len(c)) if c[i] >= t1 and c[i] <= t2]

    # MASTER VOICE OPTIMIZATION
    print('Performing master voice optimization')
    master_z = decode_audio(args.mv_seed_file, args.sample_rate)

    for iteration in range(args.n_iterations):
        print('\r> step', iteration+1, '/', args.n_iterations, end='')
        costs = []
        gradients = []
        for index, batch_sample in enumerate(random.sample(x_train, args.batch)):
            batch_z = decode_audio(batch_sample, args.sample_rate)
            input_pair = ([np.array([master_z]), np.array([batch_z])])
            cost, gradient = grab_cost_and_gradients_from_model(input_pair)
            costs.append(np.squeeze(cost))
            gradients.append(np.squeeze(gradient))
        filtered_gradients = filter_gradients(costs, gradients, args.min_sim, args.max_sim)
        perturbation = np.mean(filtered_gradients, axis=0) * args.learning_rate
        perturbation = perturbation.reshape(perturbation.shape[0], perturbation.shape[1])
        master_z += perturbation

    print('> ended optimization')