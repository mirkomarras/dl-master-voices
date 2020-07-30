#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import argparse
import os

from helpers.datapipeline import data_pipeline_verifier
from helpers.dataset import get_mv_analysis_users, load_data_set, load_test_data_from_file
from helpers.audio import load_noise_paths, cache_noise_data

from models.verifier.thinresnet34 import ThinResNet34
from models.verifier.resnet50 import ResNet50
from models.verifier.resnet34 import ResNet34
from models.verifier.xvector import XVector
from models.verifier.vggvox import VggVox

def main():
    parser = argparse.ArgumentParser(description='Speaker verifier training')

    # Parameters for a verifier
    parser.add_argument('--net', dest='net', default='', type=str, action='store', help='Network architecture')

    # Parameters for training
    parser.add_argument('--audio_dir', dest='audio_dir', default='./data/voxceleb1/dev,./data/voxceleb2/dev', type=str, action='store', help='Comma-separated audio data directories')
    parser.add_argument('--mv_data_path', dest='mv_data_path', default='./data/ad_voxceleb12/vox2_mv_data.npz', type=str, action='store', help='Numpy data for master voice analysis')
    parser.add_argument('--n_epochs', dest='n_epochs', default=1024, type=int, action='store', help='Number of epochs')
    parser.add_argument('--batch', dest='batch', default=64, type=int, action='store', help='Batch size')
    parser.add_argument('--learning_rate', dest='learning_rate', default=0.001, type=float, action='store', help='Learning rate')
    parser.add_argument('--decay_factor', dest='decay_factor', default=0.1, type=float, action='store', help='Decay factor for learning rate')
    parser.add_argument('--decay_step', dest='decay_step', default=25, type=int, action='store', help='Decay step of learning rate')
    parser.add_argument('--loss', dest='loss', default='softmax', type=str, choices=['softmax', 'amsoftmax'], action='store', help='Type of loss')
    parser.add_argument('--aggregation', dest='aggregation', default='gvlad', type=str, choices=['avg', 'vlad', 'gvlad'], action='store', help='Type of aggregation')
    parser.add_argument('--vlad_clusters', dest='vlad_clusters', default=10, type=int, action='store', help='Number of vlad clusters')
    parser.add_argument('--ghost_clusters', dest='ghost_clusters', default=2, type=int, action='store', help='Number of ghost clusters')
    parser.add_argument('--weight_decay', dest='weight_decay', default=1e-4, type=float, action='store', help='Weight decay')
    parser.add_argument('--optimizer', dest='optimizer', default='adam', choices=['adam', 'sgd'], type=str, action='store', help='Type of optimizer')
    parser.add_argument('--patience', dest='patience', default=20, type=int, action='store', help='Number of epochs with non-improving EER')
    parser.add_argument('--prefetch', dest='prefetch', default=1024, type=int, action='store', help='Number of pre-fetched batches for data pipeline')
    parser.add_argument('--embs_size', dest='embs_size', default=512, type=int, action='store', help='Size of the speaker embedding')
    parser.add_argument('--embs_name', dest='embs_name', default='embs', type=str, action='store', help='Name of the layer from which speaker embeddings are extracted')

    # Parameters for validation
    parser.add_argument('--val_base_path', dest='val_base_path', default='./data/voxceleb1/test', type=str, action='store', help='Base path for validation trials')
    parser.add_argument('--val_pair_path', dest='val_pair_path', default='./data/ad_voxceleb12/vox1_trial_pairs.csv', type=str, action='store', help='CSV file with validation trials')
    parser.add_argument('--val_n_pair', dest='val_n_pair', default=1000, type=int, action='store', help='Number of validation trials')
    parser.add_argument('--n_templates', dest='n_templates', default=1, type=int, action='store', help='Number of enrolment templates')

    # Paremeters for audio manipulation
    parser.add_argument('--sample_rate', dest='sample_rate', default=16000, type=int, action='store', help='Sample rate audio')
    parser.add_argument('--n_seconds', dest='n_seconds', default=3, type=int, action='store', help='Segment lenght in seconds')
    parser.add_argument('--augment', dest='augment', default=0, type=int, choices=[0, 1], action='store', help='Playback mode')
    parser.add_argument('--noise_dir', dest='noise_dir', default='./data/vs_noise_data', type=str, action='store', help='Noise directory')

    args = parser.parse_args()

    output_type = ('filterbank' if args.net.split('/')[0] == 'xvector' else 'spectrum')

    print('Parameters summary')

    print('>', 'Net: {}'.format(args.net))
    print('>', 'Mode: {}'.format(output_type))

    print('>', 'Val pairs dataset path: {}'.format(args.val_base_path))
    print('>', 'Val pairs path: {}'.format(args.val_pair_path))
    print('>', 'Number of val pairs: {}'.format(args.val_n_pair))

    print('>', 'Audio dir: {}'.format(args.audio_dir))
    print('>', 'Master voice data path: {}'.format(args.mv_data_path))
    print('>', 'Number of epochs: {}'.format(args.n_epochs))
    print('>', 'Batch size: {}'.format(args.batch))
    print('>', 'Learning rate: {}'.format(args.learning_rate))
    print('>', 'Decay step: {}'.format(args.decay_step))
    print('>', 'Decay factor: {}'.format(args.decay_factor))
    print('>', 'Loss: {}'.format(args.loss))
    print('>', 'Aggregation: {}'.format(args.aggregation))
    print('>', 'Optimizer: {}'.format(args.optimizer))
    print('>', 'Patience: {}'.format(args.patience))
    print('>', 'Prefetch: {}'.format(args.prefetch))
    print('>', 'Embedding size: {}'.format(args.embs_size))
    print('>', 'Embedding name: {}'.format(args.embs_name))

    print('>', 'Sample rate: {}'.format(args.sample_rate))
    print('>', 'Augmentation flag: {}'.format(args.augment))
    print('>', 'Max number of seconds: {}'.format(args.n_seconds))
    print('>', 'Noise dir: {}'.format(args.noise_dir))

    # Load noise data
    audio_dir = map(str, args.audio_dir.split(','))
    print('Load impulse response paths')
    noise_paths = load_noise_paths(args.noise_dir)
    print('Cache impulse response data')
    noise_cache = cache_noise_data(noise_paths, sample_rate=args.sample_rate)

    # Load train and validation data
    mv_user_ids = get_mv_analysis_users(args.mv_data_path)
    x_train, y_train = load_data_set(audio_dir, mv_user_ids)
    val_data = load_test_data_from_file(args.val_base_path, args.val_pair_path, args.n_templates, args.val_n_pair, args.sample_rate, args.n_seconds)
    classes = len(np.unique(y_train))

    # Data pipeline output test
    print('Checking data pipeline output')
    train_data = data_pipeline_verifier(x_train, y_train, int(args.sample_rate*args.n_seconds), args.sample_rate, args.batch, args.prefetch, output_type)

    for index, x in enumerate(train_data):
        print('>', index, x[0].shape, x[1].shape)
        if index == 10:
            break

    # Create and train model
    train_data = data_pipeline_verifier(x_train, y_train, int(args.sample_rate*args.n_seconds), args.sample_rate, args.batch, args.prefetch, output_type)

    print('Creating model')
    available_nets = {'xvector': XVector, 'vggvox': VggVox, 'resnet50': ResNet50, 'resnet34': ResNet34, 'thin_resnet': ThinResNet34}

    model = available_nets[args.net.split('/')[0]](id=(int(args.net.split('/')[1].replace('v','')) if '/v' in args.net else -1))
    model.build(classes, args.embs_size, args.embs_name, args.loss, args.aggregation, args.vlad_clusters, args.ghost_clusters, args.weight_decay, 'train')
    model.load()
    model.train(train_data, val_data, output_type, len(x_train)//args.batch, args.n_epochs, args.learning_rate, args.decay_factor, args.decay_step, args.optimizer)

if __name__ == '__main__':
    main()