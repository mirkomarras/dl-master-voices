#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import argparse
import os

from helpers.datapipeline import data_pipeline_verifier
from helpers.dataset import get_mv_analysis_users, load_data_set
from helpers.audio import load_noise_paths, cache_noise_data

from models.verifier.resnet50vox import ResNet50Vox
from models.verifier.resnet34vox import ResNet34Vox
from models.verifier.xvector import XVector
from models.verifier.vggvox import VggVox

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main():
    parser = argparse.ArgumentParser(description='Speaker verification model training')

    # Parameters for a verifier
    parser.add_argument('--net', dest='net', default='', type=str, action='store', help='Network model architecture')

    # Parameters for validation
    parser.add_argument('--val_base_path', dest='val_base_path', default='./data/voxceleb1/test', type=str, action='store', help='Base path for validation trials')
    parser.add_argument('--val_pair_path', dest='val_pair_path', default='./data/ad_voxceleb12/vox1_trial_pairs.csv', type=str, action='store', help='CSV file label, path_1, path_2 triplets')
    parser.add_argument('--val_n_pair', dest='val_n_pair', default=0, type=int, action='store', help='Number of validation pairs')

    # Parameters for training
    parser.add_argument('--audio_dir', dest='audio_dir', default='./data/voxceleb1/dev,./data/voxceleb2/dev', type=str, action='store', help='Comma-separated audio data directories')
    parser.add_argument('--mv_data_path', dest='mv_data_path', default='./data/ad_voxceleb12/vox2_mv_data.npz', type=str, action='store', help='Numpy data for master voice analysis')
    parser.add_argument('--n_epochs', dest='n_epochs', default=1024, type=int, action='store', help='Training epochs')
    parser.add_argument('--prefetch', dest='prefetch', default=100, type=int, action='store', help='Data pipeline prefetch size')
    parser.add_argument('--batch', dest='batch', default=64, type=int, action='store', help='Training batch size')
    parser.add_argument('--learning_rate', dest='learning_rate', default=0.001, type=float, action='store', help='Learning rate')
    parser.add_argument('--decay_factor', dest='decay_factor', default=0.1, type=float, action='store', help='Decay factor for learning rate')
    parser.add_argument('--decay_step', dest='decay_step', default=10, type=int, action='store', help='Decay step of learning rate')
    parser.add_argument('--loss', dest='loss', default='softmax', type=str, choices=['softmax', 'amsoftmax'], action='store', help='Type of loss')
    parser.add_argument('--aggregation', dest='aggregation', default='avg', type=str, choices=['avg', 'vlad', 'gvlad'], action='store', help='Type of aggregation')
    parser.add_argument('--vlad_clusters', dest='vlad_clusters', default=10, type=int, action='store', help='Number of vlad clusters')
    parser.add_argument('--ghost_clusters', dest='ghost_clusters', default=2, type=int, action='store', help='Number of ghost clusters')
    parser.add_argument('--weight_decay', dest='weight_decay', default=1e-4, type=float, action='store', help='Weight decay')
    parser.add_argument('--optimizer', dest='optimizer', default='adam', choices=['adam', 'sgd'], type=str, action='store', help='Type of optimizer')
    parser.add_argument('--patience', dest='patience', default=20, type=int, action='store', help='Number of epochs with non-improving EER')

    # Paremeters for raw audio
    parser.add_argument('--sample_rate', dest='sample_rate', default=16000, type=int, action='store', help='Sample rate audio')
    parser.add_argument('--n_seconds', dest='n_seconds', default=3, type=int, action='store', help='Segment lenght in seconds')
    parser.add_argument('--augment', dest='augment', default=0, type=int, choices=[0, 1], action='store', help='Data augmentation mode')
    parser.add_argument('--noise_dir', dest='noise_dir', default='./data/vs_noise_data', type=str, action='store', help='Noise directory')

    args = parser.parse_args()

    mode = ('filterbank' if args.net.split('/')[0] == 'xvector' else 'spectrum')

    print('Parameters summary')

    print('>', 'Net: {}'.format(args.net))
    print('>', 'Mode: {}'.format(mode))

    print('>', 'Val pairs dataset path: {}'.format(args.val_base_path))
    print('>', 'Val pairs path: {}'.format(args.val_pair_path))
    print('>', 'Number of val pairs: {}'.format(args.val_n_pair))

    print('>', 'Audio dir: {}'.format(args.audio_dir))
    print('>', 'Master voice data path: {}'.format(args.mv_data_path))
    print('>', 'Number of epochs: {}'.format(args.n_epochs))
    print('>', 'Prefetch: {}'.format(args.prefetch))
    print('>', 'Batch size: {}'.format(args.batch))
    print('>', 'Learning rate: {}'.format(args.learning_rate))
    print('>', 'Decay step: {}'.format(args.decay_step))
    print('>', 'Decay factor: {}'.format(args.decay_factor))
    print('>', 'Loss: {}'.format(args.loss))
    print('>', 'Aggregation: {}'.format(args.aggregation))
    print('>', 'Optimizer: {}'.format(args.optimizer))

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
    classes = len(np.unique(y_train))

    # Data pipeline output test
    print('Checking data pipeline output')
    train_data = data_pipeline_verifier(x_train, y_train, classes, sample_rate=args.sample_rate, n_seconds=args.n_seconds, batch=args.batch, prefetch=args.prefetch)

    for index, x in enumerate(train_data):
        print('>', index, x[0][0].shape, x[0][1].shape, x[1].shape)
        if index == 10:
            break

    # Create and train model
    train_data = data_pipeline_verifier(x_train, y_train, classes, sample_rate=args.sample_rate, n_seconds=args.n_seconds, batch=args.batch, prefetch=args.prefetch)

    print('Creating model')
    available_nets = {'xvector': XVector, 'vggvox': VggVox, 'resnet50vox': ResNet50Vox, 'resnet34vox': ResNet34Vox}
    model = available_nets[args.net.split('/')[0]](id=(int(args.net.split('/')[1].replace('v','')) if '/v' in args.net else -1), noises=noise_paths, cache=noise_cache, n_seconds=args.n_seconds, sample_rate=args.sample_rate)
    model.build(classes=classes, loss=args.loss, aggregation=args.aggregation, vlad_clusters=args.vlad_clusters, ghost_clusters=args.ghost_clusters, weight_decay=args.weight_decay)
    model.load()
    model.train(train_data=train_data, val_data=None, cache=noise_cache, augment=args.augment, mode=mode, steps_per_epoch=len(x_train)//args.batch, batch=args.batch, epochs=args.n_epochs, learning_rate=args.learning_rate, optimizer=args.optimizer, decay_factor=args.decay_factor, decay_step=args.decay_step)

if __name__ == '__main__':
    main()