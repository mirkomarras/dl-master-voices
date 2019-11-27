#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import librosa
import random
import time
import sys
import os

from src.helpers.datapipeline import data_pipeline_generator, data_pipeline_verifier
from src.helpers.dataset import get_mv_analysis_users, load_data_set, load_val_data
from src.helpers.audio import load_noise_paths, cache_noise_data

from src.models.verifier.tf.resnet50vox.model import ResNet50Vox
from src.models.verifier.tf.resnet34vox.model import ResNet34Vox
from src.models.verifier.tf.xvector.model import XVector
from src.models.verifier.tf.vggvox.model import VggVox

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == '__main__':

    # PARSE CLI ARGUMENTS
    parser = argparse.ArgumentParser(description='Tensorflow speaker verification model training')

    # Network architecture arguments
    parser.add_argument('--net', dest='net', default='', type=str, choices=['vggvox', 'xvector', 'resnet50vox', 'resnet34vox'], action='store', help='Network model architecture')

    # Noise and audio data arguments
    parser.add_argument('--sample_rate', dest='sample_rate', default=16000, type=int, action='store', help='Sample rate audio')
    parser.add_argument('--noise_dir', dest='noise_dir', default='./data/vs_noise_data', type=str, action='store', help='Noise directory')
    parser.add_argument('--audio_dir', dest='audio_dir', default='./data/vs_voxceleb1/dev', type=str, action='store', help='Comma-separated audio data directories')

    # Master voice train and test meta data arguments
    parser.add_argument('--mv_data_path', dest='mv_data_path', default='./data/ad_voxceleb12/vox2_mv_data.npz', type=str, action='store', help='Numpy data for master voice analysis')

    # Validation arguments
    parser.add_argument('--val_base_path', dest='val_base_path', default='./data/vs_voxceleb1/test', type=str, action='store', help='Base path for validation trials')
    parser.add_argument('--val_pair_path', dest='val_pair_path', default='./data/ad_voxceleb12/vox1_trial_pairs.csv', type=str, action='store', help='CSV file label, path_1, path_2 triplets')
    parser.add_argument('--val_n_pair', dest='val_n_pair', default=1000, type=int, action='store', help='Number of validation pairs')
    parser.add_argument('--val_interval', dest='val_interval', default=5, type=int, action='store', help='Epochs interval for computing validation')

    # Training arguments
    parser.add_argument('--n_epochs', dest='n_epochs', default=1024, type=int, action='store', help='Training epochs')
    parser.add_argument('--prefetch', dest='prefetch', default=1024, type=int, action='store', help='Data pipeline prefetch size')
    parser.add_argument('--batch', dest='batch', default=64, type=int, action='store', help='Training batch size')
    parser.add_argument('--learning_rate', dest='learning_rate', default=1e-2, type=float, action='store', help='Learning rate')
    parser.add_argument('--drop_prop', dest='drop_prop', default=0.1, type=float, action='store', help='Dropout proportion')
    parser.add_argument('--augment', dest='augment', default=0, type=int, choices=[0,1], action='store', help='Data augmentation mode')
    parser.add_argument('--n_seconds', dest='n_seconds', default=3, type=int, action='store', help='Segment lenght in seconds')
    parser.add_argument('--n_filters', dest='n_filters', default=24, type=int, action='store', help='Number of MFCCS filters')
    parser.add_argument('--buffer_size', dest='buffer_size', default=12500, type=int, action='store', help='Shuffle buffer size')

    args = parser.parse_args()

    # LOAD DATA

    # Noise data
    audio_dir = map(str, args.audio_dir.split(','))
    print('Load impulse response paths')
    noise_paths = load_noise_paths(args.noise_dir)
    print('Cache impulse response data')
    noise_cache = cache_noise_data(noise_paths, sample_rate=args.sample_rate)

    # Train data
    mv_user_ids = get_mv_analysis_users(args.mv_data_path)
    x_train, y_train = load_data_set(audio_dir, mv_user_ids)

    # CREATE DATA PIPELINE
    n_samples = 10

    # Generator output test
    print('Checking generator output')
    t1 = time.time()
    for x in data_pipeline_generator(x_train[:n_samples], y_train[:n_samples], sample_rate=args.sample_rate, n_seconds=args.n_seconds):
        t2 = time.time()
        print('>', x[0].shape, x[1], t2-t1)
        t1 = t2

    # Data pipeline output test
    print('Checking data pipeline output')

    # Load data pipeline
    iterator = data_pipeline_verifier(x_train, y_train, sample_rate=args.sample_rate, n_seconds=args.n_seconds, buffer_size=args.buffer_size, batch=args.batch, prefetch=args.prefetch)
    next_element = iterator.get_next()

    with tf.Session() as sess:
        sess.run(iterator.initializer)
        t1 = time.time()
        for i in range(n_samples):
            x = sess.run(next_element)
            t2 = time.time()
            print('>', x[0].shape, x[1][:10], t2-t1)
            t1 = t2

    # Data pipeline creation
    tf.reset_default_graph()

    # Load data pipeline
    iterator = data_pipeline_verifier(x_train, y_train, sample_rate=args.sample_rate, n_seconds=args.n_seconds, buffer_size=args.buffer_size, batch=args.batch, prefetch=args.prefetch)
    next_element = iterator.get_next()

    # LOAD VALIDATION DATA
    validation_data = load_val_data(args.val_base_path, args.val_pair_path, args.val_n_pair, args.sample_rate, args.n_seconds)

    # TRAIN AND VALIDATE MODEL
    print('Creating model')
    available_nets = {'xvector': XVector, 'vggvox': VggVox, 'resnet50vox': ResNet50Vox, 'resnet34vox': ResNet34Vox}
    model = available_nets[args.net](tf.get_default_graph())
    model.build(*next_element, len(np.unique(y_train)), n_filters=args.n_filters, noises=noise_paths, cache=noise_cache, augment=args.augment, n_seconds=args.n_seconds, sample_rate=args.sample_rate)
    model.train(n_epochs=args.n_epochs, n_steps_per_epoch=len(x_train)//args.batch, validation_data=validation_data, validation_interval=args.val_interval, learning_rate=args.learning_rate, dropout_proportion=args.drop_prop, initializer=iterator.initializer)
