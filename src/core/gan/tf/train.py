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
    parser = argparse.ArgumentParser(description='Tensorflow GAN model training')

    # Network mode and directories
    parser.add_argument('--net', dest='net', default='wavegan', type=str, choices=['wavegan'], action='store', help='Network model architecture')

    # Noise and audio data arguments
    parser.add_argument('--sample_rate', dest='sample_rate', default=16000, type=int, action='store', help='Sample rate audio')
    parser.add_argument('--audio_dir', dest='audio_dir', default='./data/vs_voxceleb1/dev', type=str, action='store', help='Comma-separated audio data directories')
    parser.add_argument('--audio_meta', dest='audio_meta', default='./data/ad_voxceleb12/vox12_meta_data.csv', type=str, action='store', help='CSV file with id-gender metadata')

    # Master voice train and test meta data arguments
    parser.add_argument('--mv_data_path', dest='mv_data_path', default='./data/ad_voxceleb12/vox2_mv_data.npz', type=str, action='store', help='Numpy data for master voice analysis')

    # Training arguments
    parser.add_argument('--batch', dest='batch', default=64, type=int, action='store', help='Training batch size')
    parser.add_argument('--save_secs', dest='save_secs', default=300, type=int, action='store', help='How often to save model')
    parser.add_argument('--summary_secs', dest='summary_secs', default=120, type=int, action='store', help='How often to report summaries')
    parser.add_argument('--gender', dest='gender', default='neutral', type=str, choices=['neutral', 'male', 'female'], action='store', help='Training gender')
    parser.add_argument('--slice_len', dest='slice_len', default=16384, type=int, choices=[16384, 32768, 65536], action='store', help='Number of audio samples per slice (maximum generation length)')
    parser.add_argument('--overlap_ratio', dest='overlap_ratio', default=0., type=float, action='store', help='Overlap ratio [0, 1) between slices')
    parser.add_argument('--first_slice', dest='first_slice', default=False, type=bool, action='store', help='If set, only use the first slice each audio example')
    parser.add_argument('--pad_end', dest='pad_end', default=False, type=bool, action='store', help='If set, use zero-padded partial slices from the end of each audio file')
    parser.add_argument('--prefetch', dest='prefetch', default=0, type=int, action='store', help='If nonnegative, prefetch examples to this GPU (Tensorflow device num)')

    # GAN arguments
    parser.add_argument('--latent_dim', dest='latent_dim', default=100, type=int, action='store', help='Number of dimensions of the latent space')
    parser.add_argument('--kernel_len', dest='kernel_len', default=25, type=int, action='store', help='Length of 1D filter kernels')
    parser.add_argument('--gan_dim', dest='gan_dim', default=64, type=int, action='store', help='Dimensionality multiplier for model of G and D')
    parser.add_argument('--use_batchnorm', dest='use_batchnorm', default=False, type=bool, action='store', help='Enable batchnorm')
    parser.add_argument('--disc_nupdates', dest='disc_nupdates', default=5, type=int, action='store', help='Number of discriminator updates per generator update')
    parser.add_argument('--loss', dest='loss', default='wgan-gp', type=str, choices=['dcgan', 'lsgan', 'wgan', 'wgan-gp'], action='store', help='Which GAN loss to use')
    parser.add_argument('--genr_upsample', dest='genr_upsample', default='zeros', type=str, choices=['zeros', 'nn'], action='store', help='Generator upsample strategy')
    parser.add_argument('--genr_pp', dest='genr_pp', default=False, action='store', help='If set, use post-processing filter')
    parser.add_argument('--genr_pp_len', dest='genr_pp_len', default=512, type=int, action='store', help='Length of post-processing filter for DCGAN')
    parser.add_argument('--disc_phaseshuffle', dest='disc_phaseshuffle', default=2, type=int, action='store', help='Radius of phase shuffle operation')

    args = parser.parse_args()

    # LOAD DATA
    audio_dir = map(str, args.audio_dir.split(','))
    mv_user_ids = get_mv_analysis_users(args.mv_data_path)
    x_train, y_train = load_data_set(audio_dir, mv_user_ids)
    x_train, y_train = filter_by_gender(x_train, y_train, args.audio_meta, args.gender)

    # GAN CREATION
    print('Creating GAN')
    available_nets = {'wavegan': WaveGAN}
    gan_model = available_nets[args.net](gender=args.gender)

    print('Training GAN')
    gan_model.infer(args.latent_dim, args.slice_len, args.kernel_len, args.gan_dim, args.use_batchnorm, args.genr_upsample, args.genr_pp, args.genr_pp_len)
    gan_model.train(x_train, args.sample_rate, args.batch, args.save_secs, args.summary_secs, args.slice_len, args.overlap_ratio, args.first_slice, args.pad_end, args.prefetch, args.latent_dim, args.kernel_len, args.gan_dim, args.use_batchnorm, args.disc_nupdates, args.loss, args.genr_upsample, args.genr_pp, args.genr_pp_len, args.disc_phaseshuffle)