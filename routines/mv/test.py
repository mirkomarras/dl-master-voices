#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from scipy.spatial.distance import cosine
from itertools import product
from loguru import logger
import tensorflow as tf
import pandas as pd
import numpy as np
import argparse

from helpers.audio import decode_audio, get_tf_spectrum, get_tf_filterbanks, get_play_n_rec_audio, load_noise_paths, cache_noise_data
from helpers.dataset import Dataset
from models import verifier

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main():
    parser = argparse.ArgumentParser(description='Tensorflow master voice trial pairs generation and testing')

    parser.add_argument('--net', dest='net', default='vggvox/v003', type=str, action='store', help='Speaker model, e.g., vggvox/v003')

    parser.add_argument('--playback', dest='playback', default=0, type=int, action='store', help='Playback and recording in master voice test condition: 0 no, 1 yes')
    parser.add_argument('--noise_dir', dest='noise_dir', default='data/vs_noise_data', type=str, action='store', help='Noise directory')

    parser.add_argument('--mv_set', dest='mv_set', default='vggvox-v000_real_f-f_mv/v000', action='store', help='Directory with MV data')
    parser.add_argument('--pop', dest='pop', default='./data/vs_mv_pairs/mv_test_population_debug_20u_10s.csv', type=str, action='store', help='Path to the filename-user_id pairs for mv training')
    parser.add_argument('--policy', dest='policy', default='any', type=str, action='store', help='Policy of verification, eigher any or avg')
    parser.add_argument('--level', dest='level', default='far1', type=str, action='store', help='Levelof security, either eer or far1')

    settings = vars(parser.parse_args())

    assert settings['net'] is not '', 'Please specify model network for --net'
    assert settings['policy'] is not '', 'Please specify policy for --policy'
    assert settings['level'] is not '', 'Please specify security level for --level'

    # Parameter summary to print at the beginning of the script
    logger.info('Parameters summary')
    for key, value in settings.items():
        logger.info(key, value)

    # Load noise data
    # print('Load impulse response paths')
    # noise_paths = load_noise_paths(args.noise_dir)
    # print('Cache impulse response data')
    # noise_cache = cache_noise_data(noise_paths)

    # Create all the paths to the mv_set/version folders we want to test
    if settings['mv_set'] is None or len(settings['mv_set']) == 0:
        mv_sets = [os.path.join('./data/vs_mv_data', mv_set, version) for mv_set in os.listdir('./data/vs_mv_data') for version in os.listdir(os.path.join('./data/vs_mv_data', mv_set))]
    else:
        mv_sets = [os.path.join('./data/vs_mv_data', settings['mv_set'])]

    logger.info(mv_sets)

    # Create the csv file with the similarity scores for each master voice, i.e., each master voice is compared with all the users enrolled templates
    logger.info('Compute similarity scores')
    for net in map(str, settings['net'].split(',')):

        # Build and load pre-trained weights of a sv
        logger.info('Loading speaker model:', net)
        sv = verifier.get_model(net)
        sv.build(classes=0, mode='test')
        sv.load()
        sv.calibrate_thresholds()
        sv.infer()

        # Create the test gallery
        test_gallery = Dataset(settings['pop'])
        test_gallery.precomputed_embeddings(sv)

        for mv_set in mv_sets:
            # TODO [Cirtical] Data not loaded correctly into predict
            # print(np.array([os.path.join(mv_set, file) for file in os.listdir(mv_set)]).shape)
            filenames = [os.path.join(mv_set, file) for file in os.listdir(mv_set) if file.endswith('.wav')]
            # logger.info(xx)
            # nn_input = np.array()
            # logger.info(f'Input size {nn_input.shape}')
            embeddings = sv.predict(filenames)
            sim_df, imp_df, gnd_df = sv.test_error_rates(embeddings, test_gallery, policy=settings['policy'], level=settings['level'], playback=None)

            # We save the similarities, impostor, and gender impostor results
            # sim_df.to_csv(os.path.join('./data/vs_mv_models/', net + '_sims_' + test_gallery.pop_file + '_' + mv_set + '_' + 'far1' + '_' + 'any'))
            # imp_df.to_csv(os.path.join('./data/vs_mv_models/', net  + '_imps_' + test_gallery.pop_file + '_' + mv_set + '_' + 'far1' + '_' + 'any'))
            # gnd_df.to_csv(os.path.join('./data/vs_mv_models/', net  + '_nds_' + test_gallery.pop_file + '_' + mv_set + '_' + 'far1' + '_' + 'any'))

if __name__ == '__main__':
    main()

