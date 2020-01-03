#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import argparse
import glob
import sys
import os

from helpers.dataset import get_mv_analysis_users, load_data_set, filter_by_gender
from helpers.datapipeline import data_pipeline_generator_gan, data_pipeline_gan
from models.gan.wavegan import WaveGAN
from models.gan.specgan import SpecGAN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == '__main__':

    # Parameters
    parser = argparse.ArgumentParser(description='Tensorflow GAN model training')

    parser.add_argument('--net', dest='net', default='wavegan', type=str, choices=['wavegan', 'specgan'], action='store', help='Network model architecture')
    parser.add_argument('--version', dest='version', default=-1, type=int, action='store', help='Version of the model to resume')
    parser.add_argument('--gender', dest='gender', default='neutral', type=str, choices=['neutral', 'male', 'female'], action='store', help='Training gender')
    parser.add_argument('--latent_dim', dest='latent_dim', default=100, type=int, action='store', help='Number of dimensions of the latent space')
    parser.add_argument('--slice_len', dest='slice_len', default=16384, type=int, choices=[16384, 32768, 65536], action='store', help='Number of dimensions of the latent space')

    parser.add_argument('--sample_rate', dest='sample_rate', default=16000, type=int, action='store', help='Sample rate audio')
    parser.add_argument('--audio_dir', dest='audio_dir', default='./data/vs_voxceleb1/dev', type=str, action='store', help='Comma-separated audio data directories')
    parser.add_argument('--audio_meta', dest='audio_meta', default='./data/ad_voxceleb12/vox12_meta_data.csv', type=str, action='store', help='CSV file with id-gender metadata')

    parser.add_argument('--mv_data_path', dest='mv_data_path', default='./data/ad_voxceleb12/vox2_mv_data.npz', type=str, action='store', help='Numpy data for master voice analysis')

    parser.add_argument('--epochs', dest='n_epochs', default=64, type=int, action='store', help='Number of epochs')
    parser.add_argument('--batch', dest='batch', default=64, type=int, action='store', help='Training batch size')
    parser.add_argument('--prefetch', dest='prefetch', default=0, type=int, action='store', help='If nonnegative, prefetch examples to this GPU (Tensorflow device num)')
    parser.add_argument('--n_seconds', dest='n_seconds', default=3, type=int, action='store', help='Segment lenght in seconds')


    args = parser.parse_args()

    print('Parameters summary')
    print('>', 'Net: {}'.format(args.net))
    print('>', 'Version: {}'.format(args.version))
    print('>', 'Gender: {}'.format(args.gender))
    print('>', 'Latent dim: {}'.format(args.latent_dim))
    print('>', 'Slice len: {}'.format(args.slice_len))
    print('>', 'Sample rate: {}'.format(args.sample_rate))
    print('>', 'Audio meta: {}'.format(args.audio_meta))
    print('>', 'Master voice data path: {}'.format(args.mv_data_path))
    print('>', 'Number of epochs: {}'.format(args.n_epochs))
    print('>', 'Batch size: {}'.format(args.batch))
    print('>', 'Prefetch: {}'.format(args.prefetch))
    print('>', 'Max number of seconds: {}'.format(args.n_seconds))

    # Load data set
    print('Loading data')
    audio_dir = map(str, args.audio_dir.split(','))
    mv_user_ids = get_mv_analysis_users(args.mv_data_path)
    x_train, y_train = load_data_set(audio_dir, mv_user_ids)
    x_train, y_train = filter_by_gender(x_train, y_train, args.audio_meta, args.gender)
    classes = len(np.unique(y_train))

    # Generator output test
    print('Checking generator output')
    for index, x in enumerate(data_pipeline_generator_gan(x_train[:10], slice_len=args.slice_len, sample_rate=args.sample_rate)):
        print('>', index, x.shape)

    # Data pipeline output test
    print('Checking data pipeline output')
    train_data = data_pipeline_gan(x_train, slice_len=args.slice_len, sample_rate=args.sample_rate, batch=args.batch, prefetch=args.prefetch)

    for index, x in enumerate(train_data):
        print('>', index, x.shape)
        if index == 10:
            break

    # Create and train model
    train_data = data_pipeline_gan(x_train, slice_len=args.slice_len, sample_rate=args.sample_rate, batch=args.batch, prefetch=args.prefetch)

    # Create GAN
    print('Creating GAN')
    available_nets = {'wavegan': WaveGAN, 'specgan': SpecGAN}
    gan_model = available_nets[args.net](name=args.net, id=args.version, gender=args.gender, latent_dim=args.latent_dim, slice_len=args.slice_len)

    # Build the model
    print('Building GAN')
    gan_model.build()

    # Train the model
    print('Training GAN')
    gan_model.train(train_data, epochs=args.n_epochs, steps_per_epoch=len(x_train)//args.batch, batch=args.batch)