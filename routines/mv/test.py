#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from scipy.spatial.distance import cosine
from itertools import product
import tensorflow as tf
import pandas as pd
import numpy as np
import argparse

from helpers.audio import decode_audio, get_tf_spectrum, get_tf_filterbanks, get_play_n_rec_audio, load_noise_paths, cache_noise_data

from models import verifier

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main():
    parser = argparse.ArgumentParser(description='Tensorflow master voice trial pairs generation and testing')

    parser.add_argument('--net', dest='net', default='', type=str, action='store', help='Speaker model, e.g., vggvox/v003')

    parser.add_argument('--playback', dest='playback', default=0, type=int, action='store', help='Playback and recording in master voice test condition: 0 no, 1 yes')
    parser.add_argument('--noise_dir', dest='noise_dir', default='data/vs_noise_data', type=str, action='store', help='Noise directory')

    parser.add_argument('--mv_set', dest='mv_set', default=None, type=str, action='append', help='Directory with MV data')
    parser.add_argument('--pop', dest='pop', default='./data/vs_mv_data/20200576-1456_mv_test_population_debug_100u_10s.csv', type=str, action='store', help='Path to the filename-user_id pairs for mv training')
    parser.add_argument('--policy', dest='policy', default='any', type=str, action='store', help='Policy of verification, eigher any or avg')
    parser.add_argument('--level', dest='level', default='far1', type=str, action='store', help='Levelof security, either eer or far1')

    settings = vars(parser.parse_args())

    assert settings['net'] is not '', 'Please specify model network for --net'
    assert settings['policy'] is not '', 'Please specify policy for --policy'
    assert settings['level'] is not '', 'Please specify security level for --level'
    assert os.path.exists(settings['pop']), 'Please specify a valid filename-user_id pairs file for mv test'

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
    if args.mv_set is None or len(args.mv_set) == 0:
        mv_sets = [os.path.join('./data/vs_mv_data', mv_set, version) for mv_set in os.listdir('./data/vs_mv_data') for version in os.listdir(os.path.join('./data/vs_mv_data', mv_set))]
    else:
        mv_sets = [os.path.join('./data/vs_mv_data', args.mv_set)]

    logger.info(mv_sets)


    # Create the csv file with the similarity scores for each master voice, i.e., each master voice is compared with all the users enrolled templates
    logger.info('Compute similarity scores')
    for net in map(str, args.net.split(',')):

        # Build and load pre-trained weights of a sv
        logger.info('Loading speaker model:', net)
        sv = verifier.get_model(net)
        sv.build(classes=0, mode='test')
        sv.load()

        # Create the test gallery
        test_gallery = Dataset(settings['pop'])
        test_gallery.precomputed_embeddings(sv)

        for mv_set in mv_sets:
            elements = tf.convert_to_tensor([os.path.join(mv_set, file) for file in os.listdir(mv_set)])
            sim_df, imp_df, gnd_df = model.test_error_rates(elements, test_gallery, policy=settings['policy'], level=settings['level'], playback=None)

            # We save the similarities, impostor, and gender impostor results
            sim_df.to_csv(os.path.join('./data/vs_mv_models/', net, 'sims_' + mv_set + '_' + settings['pop'] + '_' + settings['level'] + '_' + settings['policy']))
            imp_df.to_csv(os.path.join('./data/vs_mv_models/', net, 'imps_' + mv_set + '_' + settings['pop'] + '_' + settings['level'] + '_' + settings['policy']))
            gnd_df.to_csv(os.path.join('./data/vs_mv_models/', net, 'gnds_' + mv_set + '_' + settings['pop'] + '_' + settings['level'] + '_' + settings['policy']))


if __name__ == '__main__':
    main()

