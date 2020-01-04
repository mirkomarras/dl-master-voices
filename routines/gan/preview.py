#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import sys
import os

from helpers.dataset import get_mv_analysis_users, load_data_set, filter_by_gender
from models.gan.wavegan import WaveGAN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Tensorflow GAN model preview')

    # Parameters for a gan
    parser.add_argument('--net', dest='net', default='', type=str, action='store', help='Network model architecture')
    parser.add_argument('--gender', dest='gender', default='neutral', type=str, choices=['neutral', 'male', 'female'], action='store', help='Training gender')

    # Parameters for gan preview
    parser.add_argument('--genr_pp', dest='genr_pp', default=False, action='store', help='If set, use post-processing filter')
    parser.add_argument('--preview_n', dest='preview_n', default=32, type=int, action='store', help='Number of samples to preview')

    # Parameters for raw audio
    parser.add_argument('--sample_rate', dest='sample_rate', default=16000, type=int, action='store', help='Sample rate audio')

    args = parser.parse_args()

    print('Parameters summary')

    print('>', 'Net: {}'.format(args.net))
    print('>', 'Gender: {}'.format(args.gender))

    print('>', 'Genr pp: {}'.format(args.genr_pp))
    print('>', 'Number of preview samples: {}'.format(args.preview_n))

    print('>', 'Sample rate: {}'.format(args.sample_rate))

    assert '/v' in args.net

    # Creating a gan
    print('Creating GAN')
    available_nets = {'wavegan': WaveGAN}
    gan_model = available_nets[args.net.split('/')[0]](id=int(args.net.split('/')[1].replace('v','')), gender=args.gender)

    # Previewing a gan
    print('Previewing GAN')
    gan_model.preview(args.sample_rate, args.genr_pp, args.preview_n)
