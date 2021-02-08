#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.metrics import roc_curve
import pandas as pd
import numpy as np
import argparse
import warnings
import sys
import os

from helpers.dataset import get_mv_analysis_users, load_data_set, filter_by_gender, load_mv_data, Dataset
from helpers.datapipeline import data_pipeline_mv

from models.mv.model import SiameseModel
from models import verifier
from models import gan

logger.info('PATH>', sys.path)

warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main():
    parser = argparse.ArgumentParser(description='Master voice training')

    # Parameters for verifier
    parser.add_argument('--netv', dest='netv', default='', type=str, action='store', help='Speaker verification model, e.g., xvector/v000')

    # Parameters for generative adversarial model or of the seed voice (if netg is specified, the master voices will be created by sampling spectrums from the GAN; otherwise, you need to specify a seed voice to batch_optimize_by_path as a master voice)
    parser.add_argument('--netg', dest='netg', default=None, type=str, action='store', help='Generative adversarial model, e.g., ms-gan/v000')
    parser.add_argument('--netg_gender', dest='netg_gender', default='neutral', type=str, choices=['neutral', 'male', 'female'], action='store', help='Training gender of the generative adversarial model')
    parser.add_argument('--seed_voice', dest='seed_voice', default='', type=str, action='store', help='Path to the seed voice that will be optimized to become a master voice')

    # Parameters for master voice optimization
    parser.add_argument('--audio_dir', dest='audio_dir', default='./data/voxceleb2/dev', type=str, action='store', help='Path to the folder where master voice training audios are stored')
    parser.add_argument('--audio_meta', dest='audio_meta', default='./data/vs_mv_pairs/meta_data_vox12_all.csv', type=str, action='store', help='Path to the CSV file with id-gender metadata of master voice training audios')

    parser.add_argument('--train_pop', dest='train_pop', default='./data/vs_mv_data/20200576-1456_mv_train_population_debug_100u_10s.csv', type=str, action='store', help='Path to the filename-user_id pairs for mv training')
    parser.add_argument('--n_train_user', dest='n_train_user', default=20, type=int, action='store', help='')
    parser.add_argument('--n_test_user', dest='n_test_user', default=10, type=int, action='store', help='')
    parser.add_argument('--n_train_utterance', dest='n_train_utterance', default=20, type=int, help='')
    parser.add_argument('--n_test_utterance', dest='n_test_utterance', default=20, type=int, action='store', help='')
    parser.add_argument('--mv_gender', dest='mv_gender', default='female', type=str, choices=['neutral', 'male', 'female'], action='store', help='Geneder against which master voices will be optimized')

    parser.add_argument('--n_examples', dest='n_examples', default=100, type=int, action='store', help='Number of master voices sampled to be created (only if netg is set)')
    parser.add_argument('--n_epochs', dest='n_epochs', default=3, type=int, action='store', help='Number of optimization epochs for each master voice example')
    parser.add_argument('--batch', dest='batch', default=64, type=int, action='store', help='Size of a training batch against which master voices will be optimized')
    parser.add_argument('--prefetch', dest='prefetch', default=200, type=int, action='store', help='Number of training batches pre-fetched by the data pipeline')
    parser.add_argument('--learning_rate', dest='learning_rate', default=1e-2, type=float, action='store', help='Learning rate for master voice perturbation')
    parser.add_argument('--n_templates', dest='n_templates', default=10, type=int, action='store', help='Number of enrolment samples per user used for impersonation testing')
    parser.add_argument('--sample_rate', dest='sample_rate', default=16000, type=int, action='store', help='Sample rate of the audio files')

    parser.add_argument('--n_seconds', dest='n_seconds', default=3, type=int, action='store', help='Length in seconds of an audio for master voice optimization')
    parser.add_argument('--gradient', dest='gradient', default=None, action='store', help='Gradient mode: none / pgd / normalized')
    parser.add_argument('--max_dist', dest='max_dist', default=0, type=float, action='store', help='Max distortion (L∞)')
    parser.add_argument('--l2_reg', dest='l2_reg', default=0, type=float, action='store', help='Distortion penalty (L2 regularization)')

    parser.add_argument('--play', dest='playback', default=False, action='store_true', help='Simulate playback at optimization time')
    parser.add_argument('--ir_dir', dest='ir_dir', default='./data/vs_noise_data/', type=str, action='store', help='Path to the folder with impuse responses (room, micropone, speaker)')

    settings = vars(parser.parse_args())

    output_type = ('filterbank' if settings['netv'].split('/')[0] == 'xvector' else 'spectrum')

    assert '/v' in settings['netv'], 'The speaker verification model should be given as <model>/v<version>, e.g., vggvox/v003'

    # Parameter summary to print at the beginning of the script
    logger.info('Parameters summary')
    for key, value in settings.items():
        logger.info(key, value)

    # Load paths and labels for audio files that will be used during the optimization procedure
    if not settings['train_pop']:
        train_pop, train_set, test_set = generate_enrolled_samples(audio_meta=settings['audio_meta'], dirname='data/voxceleb2/dev', n_train=settings['n_train_user'], n_test=settings['n_test_user'], n_split=(settings['n_train_utterance'], settings['n_test_utterance']))
    else:
        train_pop, train_set, test_set = settings['train_pop'], pd.read_csv(settings['train_pop']), pd.read_csv(settings['train_pop'].replace('train', 'test'))

    x_train, y_train = train_set['filename'], train_set['user_id']
    x_train, y_train = filter_by_gender(x_train, y_train, settings['audio_meta'], settings['mv_gender'])
    assert len(x_train) > 0, 'Looks like no user data was loaded! Check your data directories and gender filters'

    logger.info('Initializing siamese network for master voice optimization')
    # Output will be saved in sub-dirs in ./data/vs_mv_data - e.g., 'vggvox-v003_real_f_sv'
    dir_name = settings['netv'].replace('/', '-') + ('_' + settings['netg'].replace('/', '-') + '_' + settings['netg_gender'][0] + '-' + settings['mv_gender'][0] if settings['netg'] else '_real_u-' + settings['mv_gender'][0])

    # We initialize the siamese model that will be used to batch_optimize_by_path master voices
    siamese_model = SiameseModel(dir=os.path.join('data', 'vs_mv_data', dir_name), params=args, playback=settings['playback'], ir_dir=settings['ir_dir'], sample_rate=settings['sample_rate'])
    logger.info('> siamese network initialized')

    logger.info('Setting verifier')
    # We initialize, build, and load a pre-trained speaker verification model; this model will be duplicated in order to create the siamese model
    sv = verifier.get_model(settings['netv'])
    siamese_model.set_verifier(sv)
    logger.info('> verifier set')

    if settings['netg'] is not None:
        logger.info('Setting generator')
        # If we want to create mv from GAN examples, we initialize, build, and load a pre-trained GAN into the siamese model
        siamese_model.set_generator(gan.get_model(settings['netg'], settings['netg_gender']))
        logger.info('> generated set')

        logger.info('Building siamese model')
    # We build a siamese model by duplicating the selected speaker verifier; hence, the siamese model will take two spectrograms as input and return the cosine similarity
    # between the speaker embeddings associated to those spectrograms. If netg is set, the right branch of the model will be fed with spectrograms generated by the GAN.
    siamese_model.build()
    logger.info('> siamese model built')

    logger.info('Checking data pipeline output')
    # We check the output of the master voice optimization pipeline, i.e., spectrograms extracted from the training audio files and their associated user labels
    train_data = data_pipeline_mv(x_train, y_train, settings['sample_rate'] * settings['n_seconds'], settings['sample_rate'], settings['batch'], settings['prefetch'], output_type)

    for index, x in enumerate(train_data):
        logger.info('  ', index, x[0].shape, x[1].shape)
        if index == 2:
            break

    # Setup training and testing datasets
    train_data = data_pipeline_mv(x_train, y_train, settings['sample_rate'] * settings['n_seconds'], settings['sample_rate'], settings['batch'], settings['prefetch'], output_type)
    test_gallery = Dataset(train_pop)
    test_gallery.precomputed_embeddings(sv)

    # Construct optimization settings
    opt_settings = siamese_model.defaults().update({
        'gradient': settings['gradient'],
        'n_epochs': settings['n_epochs'],
        'max_perturbation': settings['max_dist'],
        'l2_regularization': settings['l2_reg'],
        'n_examples': settings['n_examples']
    })

    # Run optimization
    siamese_model.batch_optimize_by_path(settings['seed_voice'], train_data, test_gallery, settings=opt_settings)

if __name__ == '__main__':
    main()
