#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import argparse
import os

from helpers.datapipeline import data_pipeline_generator_mv, data_pipeline_mv
from helpers.dataset import get_mv_analysis_users, load_data_set, filter_by_gender, load_test_data_from_file, create_template_trials, load_mv_data
from models.verifier.resnet50vox import ResNet50Vox
from models.verifier.resnet34vox import ResNet34Vox
from models.verifier.xvector import XVector
from models.verifier.vggvox import VggVox
from models.gan.wavegan import WaveGAN
from models.gan.specgan import SpecGAN
from models.mv.model import MasterVocoder

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main():
    parser = argparse.ArgumentParser(description='Master voice training')

    # Parameters for verifier
    parser.add_argument('--net_verifier', dest='net_verifier', default='', type=str, action='store', help='Network model architecture e.g., xvector/v0')
    parser.add_argument('--classes', dest='classes', default=5205, type=int, action='store', help='Classes')
    parser.add_argument('--policy', dest='policy', default='any', type=str, action='store', help='Verification policy')
    parser.add_argument('--n_templates', dest='n_templates', default=1, type=int, action='store', help='Number of enrolment templates')

    # Parameters for gan
    parser.add_argument('--net_gan', dest='net_gan', default='', type=str, action='store', help='Network model architecture e.g. wavegan/female/v0')
    parser.add_argument('--mv_input_path', dest='mv_input_path', default='', type=str, action='store', help='Base master voice path for spectrum optimization')
    parser.add_argument('--gender_gan', dest='gender_gan', default='female', type=str, choices=['neutral', 'male', 'female'], action='store', help='Training gender')
    parser.add_argument('--latent_dim', dest='latent_dim', default=100, type=int, action='store', help='Number of dimensions of the latent space')
    parser.add_argument('--slice_len', dest='slice_len', default=16384, type=int, choices=[16384, 32768, 65536], action='store', help='Number of dimensions of the latent space')

    # Parameters for training
    parser.add_argument('--audio_dir', dest='audio_dir', default='./data/voxceleb1/dev', type=str, action='store', help='Comma-separated audio data directories')
    parser.add_argument('--audio_meta', dest='audio_meta', default='./data/ad_voxceleb12/vox12_meta_data.csv', type=str, action='store', help='CSV file with id-gender metadata')
    parser.add_argument('--gender_train', dest='gender_train', default='female', type=str, choices=['neutral', 'male', 'female'], action='store', help='Training gender')
    parser.add_argument('--n_iterations', dest='n_iterations', default=100, type=int, action='store', help='Number of iterations')
    parser.add_argument('--n_epochs', dest='n_epochs', default=1024, type=int, action='store', help='Training epochs')
    parser.add_argument('--prefetch', dest='prefetch', default=1024, type=int, action='store', help='Data pipeline prefetch size')
    parser.add_argument('--batch', dest='batch', default=32, type=int, action='store', help='Training batch size')
    parser.add_argument('--learning_rate', dest='learning_rate', default=1e-2, type=float, action='store', help='Learning rate')

    # Parameters for testing verifier against eer
    parser.add_argument('--sv_base_path', dest='sv_base_path', default='./data/voxceleb1/test', type=str, action='store', help='Trials base path for computing speaker verification thresholds')
    parser.add_argument('--sv_pair_path', dest='sv_pair_path', default='./data/ad_voxceleb12/vox1_trial_pairs.csv', type=str, action='store', help='CSV file label, path_1, path_2 speaker verification trials')
    parser.add_argument('--sv_n_pair', dest='sv_n_pair', default=0, type=int, action='store', help='Number of speaker verification trials')

    # Parameters for master voice analysis
    parser.add_argument('--mv_base_path', dest='mv_base_path', default='./data/voxceleb2/', type=str, action='store', help='Trials base path for master voice analysis waveforms')
    parser.add_argument('--mv_meta', dest='mv_meta', default='./data/ad_voxceleb12/vox2_mv_data.npz', type=str, action='store', help='Numpy data for master voice analysis metadata')
    parser.add_argument('--n_templates', dest='n_templates', type=int, default=10, action='store', help='Enrolment set size')

    # Parameters for raw audio
    parser.add_argument('--n_seconds', dest='n_seconds', default=3, type=int, action='store', help='Segment lenght in seconds')
    parser.add_argument('--sample_rate', dest='sample_rate', default=16000, type=int, action='store', help='Sample rate audio')

    args = parser.parse_args()

    mode = ('filterbank' if args.net_verifier.split('/')[0] == 'xvector' else 'spectrum')

    print('Parameters summary')

    print('>', 'Net Verifier: {}'.format(args.net_verifier))
    print('>', 'Policy: {}'.format(args.policy))
    print('>', 'Number of enrolment templates: {}'.format(args.n_templates))
    print('>', 'Mode: {}'.format(mode))

    print('>', 'Net GAN: {}'.format(args.net_gan))
    print('>', 'Gender GAN: {}'.format(args.gender_gan))
    print('>', 'Latent dim: {}'.format(args.latent_dim))
    print('>', 'Slice len: {}'.format(args.slice_len))

    print('>', 'Master voice input path: {}'.format(args.mv_input_path))
    print('>', 'Train audio dirs: {}'.format(args.audio_dir))
    print('>', 'Train audio meta path: {}'.format(args.audio_meta))
    print('>', 'Gender train: {}'.format(args.gender_train))
    print('>', 'Number of iterations: {}'.format(args.n_iterations))
    print('>', 'Number of epochs: {}'.format(args.n_epochs))
    print('>', 'Prefetch: {}'.format(args.prefetch))
    print('>', 'Batch size: {}'.format(args.batch))
    print('>', 'Learning rate: {}'.format(args.learning_rate))

    print('>', 'Test pairs path: {}'.format(args.sv_pair_path))
    print('>', 'Number of test pairs: {}'.format(args.sv_n_pair))
    print('>', 'Test dataset path: {}'.format(args.sv_base_path))

    print('>', 'Master voice base path: {}'.format(args.mv_base_path))
    print('>', 'Master voice meta path: {}'.format(args.mv_meta))
    print('>', 'Number of samples per template: {}'.format(args.n_templates))

    print('>', 'Sample rate: {}'.format(args.sample_rate))
    print('>', 'Max number of seconds: {}'.format(args.n_seconds))

    assert '/v' in args.net_verifier and '/v' in args.net_gan

    audio_dir = map(str, args.audio_dir.split(','))
    mv_user_ids = get_mv_analysis_users(args.mv_meta, type='train')
    x_train, y_train = load_data_set(audio_dir, mv_user_ids, include=True)
    x_train, y_train = filter_by_gender(x_train, y_train, args.audio_meta, args.gender_train)

    print('Initializing vocoder')
    dir_mv = os.path.join('.', 'data', 'vs_mv_data', args.net_verifier.replace('/', '-') + '_' + args.net_gan.replace('/', '-') + '_' + args.gender_gan[0] + '-' + args.gender_train[0] + '_mv')
    dir_sv = os.path.join('.', 'data', 'vs_mv_data', args.net_verifier.replace('/', '-') + '_' + args.net_gan.replace('/', '-') + '_' + args.gender_gan[0] + '-' + args.gender_train[0] + '_sv')
    vocoder = MasterVocoder(sample_rate=args.sample_rate, dir_mv=dir_mv, dir_sv=dir_sv)
    print('> vocoder initialized')

    print('Setting verifier')
    available_verifiers = {'xvector': XVector, 'vggvox': VggVox, 'resnet50vox': ResNet50Vox, 'resnet34vox': ResNet34Vox}
    selected_verifier = available_verifiers[args.net_verifier.split('/')[0]](id=int(args.net_verifier.split('/')[1].replace('v','')), n_seconds=args.n_seconds, sample_rate=args.sample_rate)
    vocoder.set_verifier(selected_verifier, args.classes)
    print('> verifier set')

    if args.net_gan is not None:
        print('Setting generator')
        available_generators = {'wavegan': WaveGAN, 'specgan': SpecGAN}
        selected_generator = available_generators[args.net_gan.split('/')[0]](id=int(args.net_gan.split('/')[1].replace('v','')), gender=args.gender_gan, latent_dim=args.latent_dim, slice_len=args.slice_len)
        vocoder.set_generator(selected_generator)
        print('> generated set')

    print('Setting learning phase')
    tf.keras.backend.set_learning_phase(0)
    print('> learning phase set', tf.keras.backend.learning_phase())

    print('Building vocoder')
    vocoder.build(mode=mode)
    print('> vocoder built')

    print('Checking generator output')
    for index, x in enumerate(data_pipeline_generator_mv(x_train[:10], sample_rate=args.sample_rate, n_seconds=args.n_seconds)):
        print('>', index, x.shape),

    print('Checking data pipeline output')
    train_data = data_pipeline_mv(x_train, sample_rate=args.sample_rate, n_seconds=args.n_seconds, batch=args.batch, prefetch=args.prefetch)
    for index, x in enumerate(train_data):
        print('>', index, x.shape)
        if index == 10:
            break

    if not os.path.exists(args.sv_pair_path):
        print('Creating trials file with templates', args.n_templates)
        create_template_trials(args.sv_base_path, args.sv_pair_path, args.n_templates, args.sv_n_pair, args.sv_n_pair)
        print('> trial pairs file saved')

    print('Loading data for training and testing master voice impersonation')
    test_data = load_test_data_from_file(args.sv_base_path, args.sv_pair_path, args.sv_n_pair, args.sample_rate, args.n_seconds)
    mv_test_thrs = selected_verifier.test(test_data, mode, args.policy)
    mv_test_data = load_mv_data(args.mv_meta, args.mv_base_path, args.audio_meta, args.sample_rate, args.n_seconds, args.n_templates)

    mv_train_data = data_pipeline_mv(x_train, sample_rate=args.sample_rate, n_seconds=args.n_seconds, batch=args.batch, prefetch=args.prefetch)

    print('Optimizing master voice')
    vocoder.train(args.mv_input_path, mv_train_data, args.n_iterations, args.n_epochs, len(x_train) // args.batch, mv_test_thrs, mv_test_data)

if __name__ == '__main__':
    main()