#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import sys
import os

from src.helpers.dataset import get_mv_analysis_users, load_data_set, filter_by_gender
from src.models.gan.tf.wavegan.model import WaveGAN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == '__main__':

    # PARSE CLI ARGUMENTS
    parser = argparse.ArgumentParser(description='Tensorflow GAN model preview')

    # Network mode and directories
    parser.add_argument('--net', dest='net', default='wavegan', type=str, choices=['wavegan'], action='store', help='Network model architecture')
    parser.add_argument('--version', dest='version', default='', type=str, action='store', help='Network version number')
    parser.add_argument('--gender', dest='gender', default='neutral', type=str, choices=['neutral', 'male', 'female'], action='store', help='Training gender')

    # Noise and audio data arguments
    parser.add_argument('--sample_rate', dest='sample_rate', default=16000, type=int, action='store', help='Sample rate audio')

    # GAN arguments
    parser.add_argument('--genr_pp', dest='genr_pp', default=False, action='store', help='If set, use post-processing filter')
    parser.add_argument('--preview_n', dest='preview_n', default=32, type=int, action='store', help='Number of samples to preview')

    args = parser.parse_args()

    # GAN CREATION
    print('Creating GAN')
    available_nets = {'wavegan': WaveGAN}
    gan_model = available_nets[args.net](id=args.version, gender=args.gender)

    print('Previewing GAN')
    gan_model.preview(args.sample_rate, args.genr_pp, args.preview_n)
