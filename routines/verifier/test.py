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

from helpers.dataset import load_test_data

from models.verifier.vggvox import VggVox
from models.verifier.xvector import XVector
from models.verifier.resnet50vox import ResNet50Vox
from models.verifier.resnet34vox import ResNet34Vox

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main():

    parser = argparse.ArgumentParser(description='Tensorflow speaker verification model training')

    # Parameters
    parser.add_argument('--net', dest='net', default='', type=str, choices=['vggvox', 'xvector', 'resnet50vox', 'resnet34vox'], action='store', help='Network model architecture')
    parser.add_argument('--version', dest='version', default='', type=str, action='store', help='Network version number')

    parser.add_argument('--test_base_path', dest='test_base_path', default='./data/vs_voxceleb1/test', type=str, action='store', help='Base path for validation trials')
    parser.add_argument('--test_pair_path', dest='test_pair_path', default='./data/ad_voxceleb12/vox1_trial_pairs.csv', type=str, action='store', help='CSV file label, path_1, path_2 triplets')
    parser.add_argument('--test_n_pair', dest='test_n_pair', default=0, type=int, action='store', help='Number of test pairs')

    parser.add_argument('--sample_rate', dest='sample_rate', default=16000, type=int, action='store', help='Sample rate audio')
    parser.add_argument('--n_seconds', dest='n_seconds', default=3, type=int, action='store', help='Segment lenght in seconds')

    args = parser.parse_args()

    print('Parameters summary')
    print('>', 'Net: {}'.format(args.net))
    print('>', 'Version: {}'.format(args.version))
    print('>', 'Sample rate: {}'.format(args.sample_rate))
    print('>', 'Test pairs dataset path: {}'.format(args.test_base_path))
    print('>', 'Test pairs path: {}'.format(args.test_pair_path))
    print('>', 'Number of test pairs: {}'.format(args.test_n_pair))
    print('>', 'Max number of seconds: {}'.format(args.n_seconds))

    # Load test data
    test_data = load_test_data(args.test_base_path, args.test_pair_path, args.test_n_pair, args.sample_rate, args.n_seconds)

    # Create model
    print('Creating model')
    available_nets = {'xvector': XVector, 'vggvox': VggVox, 'resnet50vox': ResNet50Vox, 'resnet34vox': ResNet34Vox}
    model = available_nets[args.net](id=args.version, n_seconds=args.n_seconds, sample_rate=args.sample_rate)

    # Test model
    print('Testing model')
    t1 = time.time()
    model.test(test_data)
    t2 = time.time()
    print('>', t2-t1, 'seconds for testing')

if __name__ == '__main__':
    main()