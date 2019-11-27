#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import random
import os

from src.helpers.audio import load_noise_paths, cache_noise_data, decode_audio
from src.helpers.dataset import load_test_data, load_mv_data

from src.models.verifier.tf.resnet50vox.model import ResNet50Vox
from src.models.verifier.tf.resnet34vox.model import ResNet34Vox
from src.models.verifier.tf.xvector.model import XVector
from src.models.verifier.tf.vggvox.model import VggVox

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def print_results(results, mv_type):
    eer_any_m = np.mean([results[mv_type][f]['any']['eer']['m'] for f in results[mv_type].keys()])
    eer_any_f = np.mean([results[mv_type][f]['any']['eer']['f'] for f in results[mv_type].keys()])
    eer_avg_m = np.mean([results[mv_type][f]['avg']['eer']['m'] for f in results[mv_type].keys()])
    eer_avg_f = np.mean([results[mv_type][f]['avg']['eer']['f'] for f in results[mv_type].keys()])
    far1_any_m = np.mean([results[mv_type][f]['any']['far1']['m'] for f in results[mv_type].keys()])
    far1_any_f = np.mean([results[mv_type][f]['any']['far1']['f'] for f in results[mv_type].keys()])
    far1_avg_m = np.mean([results[mv_type][f]['avg']['far1']['m'] for f in results[mv_type].keys()])
    far1_avg_f = np.mean([results[mv_type][f]['avg']['far1']['f'] for f in results[mv_type].keys()])
    print("\r{:<15}".format(mv_type), end=' ')
    print("%0.3f %0.3f %0.3f %0.3f %0.3f %0.3f %0.3f %0.3f" % (eer_any_m, eer_any_f, eer_avg_m, eer_avg_f, far1_any_m, far1_any_f, far1_avg_m, far1_avg_f), end='')

if __name__ == '__main__':

    # PARSE CLI ARGUMENTS
    parser = argparse.ArgumentParser(description='Tensorflow speaker verification model training')

    # Network architecture arguments
    parser.add_argument('--net', dest='net', default='', type=str, choices=['vggvox', 'xvector', 'resnet50vox', 'resnet34vox'], action='store', help='Network model architecture')
    parser.add_argument('--net_version', dest='net_version', default='', type=str, action='store', help='Network version number')

    # Noise and audio data arguments
    parser.add_argument('--sample_rate', dest='sample_rate', default=16000, type=int, action='store', help='Sample rate audio')
    parser.add_argument('--n_seconds', dest='n_seconds', default=3, type=int, action='store', help='Segment lenght in seconds')
    parser.add_argument('--augment', dest='augment', default=0, type=int, choices=[0,1], action='store', help='Data augmentation mode')
    parser.add_argument('--noise_dir', dest='noise_dir', default='./data/vs_noise_data', type=str, action='store', help='Noise directory')
    parser.add_argument('--audio_meta', dest='audio_meta', default='./data/ad_voxceleb12/vox12_meta_data.csv', type=str, action='store', help='CSV file with id-gender metadata')

    # Test arguments
    parser.add_argument('--sv_base_path', dest='sv_base_path', default='./data/vs_voxceleb1/test', type=str, action='store', help='Trials base path for computing speaker verification thresholds')
    parser.add_argument('--sv_pair_path', dest='sv_pair_path', default='./data/ad_voxceleb12/vox1_trial_pairs.csv', type=str, action='store', help='CSV file label, path_1, path_2 speaker verification trials')
    parser.add_argument('--sv_n_pair', dest='sv_n_pair', default=1000, type=int, action='store', help='Number of speaker verification trials')

    # Master voice train and test meta data arguments
    parser.add_argument('--mv_base_path', dest='mv_base_path', default='./data/vs_voxceleb2/', type=str, action='store', help='Trials base path for master voice analysis waveforms')
    parser.add_argument('--mv_meta', dest='mv_meta', default='./data/ad_voxceleb12/vox2_mv_data.npz', type=str, action='store', help='Numpy data for master voice analysis metadata')
    parser.add_argument('--mv_set_path', dest='mv_set_path', default='./data/vs_mv_data', type=str, action='store', help='Master voice populations directory')
    parser.add_argument('--mv_set_version', dest='mv_set_version', default='', type=str, action='store', help='Master voice populations version number')

    # Enrolment and verification parameters
    parser.add_argument('--n_templates', dest='n_templates', type=int, default=10, action='store', help='Enrolment set size')

    args = parser.parse_args()

    # LOAD NOISE DATA
    print('Load impulse response paths')
    noise_paths = load_noise_paths(args.noise_dir)
    print('Cache impulse response data')
    noise_cache = cache_noise_data(noise_paths, sample_rate=args.sample_rate)

    # RESTORE MODEL
    print('Creating model')
    available_nets = {'xvector': XVector, 'vggvox': VggVox, 'resnet50vox': ResNet50Vox, 'resnet34vox': ResNet34Vox}
    model = available_nets[args.net](tf.get_default_graph(), reuse=True, id=args.net_version)
    model.build(None, None, noises=noise_paths, cache=noise_cache, augment=args.augment)

    # RETRIEVE THRESHOLDS
    (x1_test, x2_test), y_test = load_test_data(args.sv_base_path, args.sv_pair_path, args.sv_n_pair, args.sample_rate, args.n_seconds)
    thr_eer, thr_far1 = model.test(((x1_test, x2_test), y_test))

    # LOAD DATA FOR IMPERSONATION TEST
    x_mv_test, y_mv_test, male_x_mv_test, female_x_mv_test = load_mv_data(args.mv_meta, args.mv_base_path, args.audio_meta, args.sample_rate, args.n_seconds, args.n_templates)

    # TEST MODEL AGAINST MASTER VOICE IMPERSONATION: {population_name, mv_file, {any/avg}, {eer/far1}, {m/f}}
    print('Testing master voice impersonation')
    print("{:<15} {:<23} {:<23}".format('', 'EER', 'FAR1%'))
    print("{:<15} {:<11} {:<11} {:<11} {:<11}".format('', 'ANY', 'AVG', 'ANY', 'AVG'))
    print("{:<15} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5}".format('', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F'))

    results = {}
    for mv_type_index, mv_type in enumerate(os.listdir(args.mv_set_path)):

        results[mv_type] = {}

        for mv_file_index, mv_file in enumerate(os.listdir(os.path.join(args.mv_set_path, mv_type, 'v' + args.mv_set_version))):

            if mv_file.endswith('.wav'):

                results[mv_type][mv_file] = {}
                mv_data = decode_audio(os.path.join(args.mv_set_path, mv_type, 'v' + args.mv_set_version, mv_file), sample_rate=args.sample_rate).reshape((-1, 1))[:args.sample_rate*args.n_seconds, :]

                for policy in ['any', 'avg']:
                    results[mv_type][mv_file][policy] = {}
                    for thr_type, thr in {'eer': thr_eer, 'far1': thr_far1}.items():
                        results[mv_type][mv_file][policy][thr_type] = model.impersonate(mv_data, thr, policy, x_mv_test, y_mv_test, male_x_mv_test, female_x_mv_test, args.n_templates)

                print_results(results, mv_type)

        print()