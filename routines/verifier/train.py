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

from helpers.datapipeline import data_pipeline_generator, data_pipeline_verifier
from helpers.dataset import get_mv_analysis_users, load_data_set, load_val_data
from helpers.audio import load_noise_paths, cache_noise_data

from models.verifier.resnet50vox import ResNet50Vox
from models.verifier.resnet34vox import ResNet34Vox
from models.verifier.xvector import XVector
from models.verifier.vggvox import VggVox

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main():
    parser = argparse.ArgumentParser(description='Speaker verification model training')

    # Parameters
    parser.add_argument('--net', dest='net', default='', type=str, choices=['vggvox', 'xvector', 'resnet50vox', 'resnet34vox'], action='store', help='Network model architecture')
    parser.add_argument('--version', dest='version', default=-1, type=int, action='store', help='Version of the model to resume')

    parser.add_argument('--sample_rate', dest='sample_rate', default=16000, type=int, action='store', help='Sample rate audio')
    parser.add_argument('--noise_dir', dest='noise_dir', default='./data/vs_noise_data', type=str, action='store', help='Noise directory')
    parser.add_argument('--audio_dir', dest='audio_dir', default='./data/vs_voxceleb1/dev', type=str, action='store', help='Comma-separated audio data directories')

    parser.add_argument('--mv_data_path', dest='mv_data_path', default='./data/ad_voxceleb12/vox2_mv_data.npz', type=str, action='store', help='Numpy data for master voice analysis')

    parser.add_argument('--val_base_path', dest='val_base_path', default='./data/vs_voxceleb1/test', type=str, action='store', help='Base path for validation trials')
    parser.add_argument('--val_pair_path', dest='val_pair_path', default='./data/ad_voxceleb12/vox1_trial_pairs.csv', type=str, action='store', help='CSV file label, path_1, path_2 triplets')
    parser.add_argument('--val_n_pair', dest='val_n_pair', default=10, type=int, action='store', help='Number of validation pairs')

    parser.add_argument('--n_epochs', dest='n_epochs', default=1024, type=int, action='store', help='Training epochs')
    parser.add_argument('--prefetch', dest='prefetch', default=1024, type=int, action='store', help='Data pipeline prefetch size')
    parser.add_argument('--batch', dest='batch', default=64, type=int, action='store', help='Training batch size')
    parser.add_argument('--learning_rate', dest='learning_rate', default=1e-2, type=float, action='store', help='Learning rate')
    parser.add_argument('--augment', dest='augment', default=0, type=int, choices=[0,1], action='store', help='Data augmentation mode')
    parser.add_argument('--n_seconds', dest='n_seconds', default=3, type=int, action='store', help='Segment lenght in seconds')
    parser.add_argument('--buffer_size', dest='buffer_size', default=12500, type=int, action='store', help='Shuffle buffer size')

    args = parser.parse_args()

    print('Parameters summary')
    print('>', 'Net: {}'.format(args.net))
    print('>', 'Version: {}'.format(args.version))
    print('>', 'Sample rate: {}'.format(args.sample_rate))
    print('>', 'Noise dir: {}'.format(args.noise_dir))
    print('>', 'Audio dir: {}'.format(args.audio_dir))
    print('>', 'Master voice data path: {}'.format(args.mv_data_path))
    print('>', 'Val pairs dataset path: {}'.format(args.val_base_path))
    print('>', 'Val pairs path: {}'.format(args.val_pair_path))
    print('>', 'Number of val pairs: {}'.format(args.val_n_pair))
    print('>', 'Number of epochs: {}'.format(args.n_epochs))
    print('>', 'Batch size: {}'.format(args.batch))
    print('>', 'Augmentation flag: {}'.format(args.augment))
    print('>', 'Prefetch: {}'.format(args.prefetch))
    print('>', 'Buffer size: {}'.format(args.buffer_size))
    print('>', 'Max number of seconds: {}'.format(args.n_seconds))

    # Load noise data
    audio_dir = map(str, args.audio_dir.split(','))
    print('Load impulse response paths')
    noise_paths = load_noise_paths(args.noise_dir)
    print('Cache impulse response data')
    noise_cache = cache_noise_data(noise_paths, sample_rate=args.sample_rate)

    # Load train and validation data
    mv_user_ids = get_mv_analysis_users(args.mv_data_path)
    x_train, y_train = load_data_set(audio_dir, mv_user_ids)
    val_data = load_val_data(args.val_base_path, args.val_pair_path, args.val_n_pair, args.sample_rate, args.n_seconds)
    classes = len(np.unique(y_train))

    # Generator output test
    print('Checking generator output')
    for index, x in enumerate(data_pipeline_generator(x_train[:10], y_train[:10], classes, augment=args.augment, sample_rate=args.sample_rate, n_seconds=args.n_seconds)):
        print('>', index, x[0]['input_1'].shape, x[0]['input_2'].shape, x[1].shape)

    # Data pipeline output test
    print('Checking data pipeline output')
    train_data = data_pipeline_verifier(x_train, y_train, classes, augment=args.augment, sample_rate=args.sample_rate, n_seconds=args.n_seconds, buffer_size=args.buffer_size, batch=args.batch, prefetch=args.prefetch)

    for index, x in enumerate(train_data):
        print('>', index, x[0]['input_1'].shape, x[0]['input_2'].shape, x[1].shape)
        if index == 10:
            break

    # Create and train model
    train_data = data_pipeline_verifier(x_train, y_train, classes, augment=args.augment, sample_rate=args.sample_rate, n_seconds=args.n_seconds, buffer_size=args.buffer_size, batch=args.batch, prefetch=args.prefetch)
    print('Creating model')
    available_nets = {'xvector': XVector, 'vggvox': VggVox, 'resnet50vox': ResNet50Vox, 'resnet34vox': ResNet34Vox}
    model = available_nets[args.net](id=args.version, noises=noise_paths, cache=noise_cache, n_seconds=args.n_seconds, sample_rate=args.sample_rate)
    model.build(classes=classes)
    model.train(train_data, val_data, steps_per_epoch=len(x_train)//args.batch, epochs=args.n_epochs, learning_rate=args.learning_rate)

if __name__ == '__main__':
    main()