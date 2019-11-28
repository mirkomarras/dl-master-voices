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

from src.helpers.dataset import load_test_data

from src.models.verifier.tf.vggvox.model import VggVox
from src.models.verifier.tf.xvector.model import XVector
from src.models.verifier.tf.resnet50vox.model import ResNet50Vox
from src.models.verifier.tf.resnet34vox.model import ResNet34Vox

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == '__main__':

    # PARSE CLI ARGUMENTS
    parser = argparse.ArgumentParser(description='Tensorflow speaker verification model training')

    # Network architecture arguments
    parser.add_argument('--net', dest='net', default='', type=str, choices=['vggvox', 'xvector', 'resnet50vox', 'resnet34vox'], action='store', help='Network model architecture')
    parser.add_argument('--version', dest='version', default='', type=str, action='store', help='Network version number')

    # Test arguments
    parser.add_argument('--test_base_path', dest='test_base_path', default='./data/vs_voxceleb1/test', type=str, action='store', help='Base path for validation trials')
    parser.add_argument('--test_pair_path', dest='test_pair_path', default='./data/ad_voxceleb12/vox1_trial_pairs.csv', type=str, action='store', help='CSV file label, path_1, path_2 triplets')
    parser.add_argument('--test_n_pair', dest='test_n_pair', default=0, type=int, action='store', help='Number of test pairs')

    # Training arguments
    parser.add_argument('--sample_rate', dest='sample_rate', default=16000, type=int, action='store', help='Sample rate audio')
    parser.add_argument('--n_filters', dest='n_filters', default=24, type=int, action='store', help='Number of MEL bins')
    parser.add_argument('--n_seconds', dest='n_seconds', default=3, type=int, action='store', help='Segment lenght in seconds')

    args = parser.parse_args()

    # LOAD TEST DATA
    (x1_test, x2_test), y_test = load_test_data(args.test_base_path, args.test_pair_path, args.test_n_pair, args.sample_rate, args.n_seconds)

    # TRAIN AND VALIDATE MODEL
    print('Creating model')
    available_nets = {'xvector': XVector, 'vggvox': VggVox, 'resnet50vox': ResNet50Vox, 'resnet34vox': ResNet34Vox}
    model = available_nets[args.net](tf.get_default_graph(), reuse=True, id=args.version)
    model.build(None, None)

    t1 = time.time()
    model.test(((x1_test, x2_test), y_test))
    t2 = time.time()
    print('>', t2-t1, 'seconds for testing', len(y_test), 'pairs')
