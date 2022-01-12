#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from genericpath import exists
import itertools
import os
import sys
import glob 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from loguru import logger
import numpy as np
import argparse
import json
from collections import defaultdict

import tensorflow as tf
from helpers import utils
from helpers.dataset import Dataset
from models import verifier

utils.setup_logging(level='DEBUG')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main():
    parser = argparse.ArgumentParser(description='Tensorflow master voice trial pairs generation and testing')

    parser.add_argument('--net', dest='net', default='vggvox/v000,resnet50/v000,thin_resnet/v000,resnet34/v000', type=str, action='store', help='Speaker model, e.g., vggvox/v003') #vggvox/v004,resnet50/v004,thin_resnet/v002,resnet34/v002,xvector/v003
    parser.add_argument('--samples', dest='mv_set', required=True, action='store', help='Directory with speech samples')
    parser.add_argument('--dataset', dest='dataset', default='dev-test', type=str, action='store', help='JSON file with population settings (quick setup)')
    parser.add_argument('--pop', dest='pop', default='data/vs_mv_pairs/mv_test_population_interspeech_1000u_10s.csv', type=str, action='store', help='Path to the filename-user_id pairs for mv training')
    parser.add_argument('--playback', dest='playback', default='0,1', action='store', help='Playback and recording in master voice test condition: 0 no, 1 yes')
    parser.add_argument('--noise_dir', dest='noise_dir', default='data/vs_noise_data', type=str, action='store', help='Noise directory')
    parser.add_argument('--policy', dest='policy', default='any,avg', type=str, action='store', help='Policy of verification, eigher any or avg')
    parser.add_argument('--level', dest='level', default='eer,far1', type=str, action='store', help='Levelof security, either eer or far1')
    parser.add_argument('--n_play_reps', dest='n_play_reps', default=20, type=int, action='store', help='Number of randomized playback settings to run the test for')
    parser.add_argument('--impulse_flags', dest='impulse_flags', nargs='+', help='Impulse Flags for controlling playback', required=False)
    parser.add_argument('--roll', dest='roll', default=None, type=int, action='store', help='Length to rolling the audio sample')
    parser.add_argument('--train_analysis', dest='train_analysis', default=False, type=bool, action='store', help='Picking filenames for train analysis')
    parser.add_argument('--memory-growth', dest='memory_growth', action='store_true', help='Enable dynamic memory growth in Tensorflow')                       

    args = parser.parse_args()

    if args.dataset:
        if '/' in args.dataset:
            subset = args.dataset.split('/')[-1]
            dataset_label = args.dataset.split('/')[0]
        else:
            subset = 'test'
            dataset_label = args.dataset.split('/')[0]
        with open('config/datasets.json') as f:
            data_config = defaultdict()
            data_config.update(json.load(f)[dataset_label])
        args.pop = data_config[subset]

    if args.impulse_flags is None:
        args.impulse_flags = (1, 1, 1)
    else:
        args.impulse_flags = ','.join(args.impulse_flags)
        args.impulse_flags = tuple(int(x) for x in args.impulse_flags.split(','))
    assert len(args.impulse_flags) == 3

    settings = vars(args)

    assert settings['net'] != '', 'Please specify model network for --net'
    assert settings['policy'] != '', 'Please specify policy for --policy'
    assert settings['level'] != '', 'Please specify security level for --level'

    # Parameter summary to print at the beginning of the script
    logger.info('Parameters summary')
    for key, value in settings.items():
        logger.debug(f'  {key} = {value}')

    # Load noise data
    # print('Load impulse response paths')
    # noise_paths = load_noise_paths(args.noise_dir)
    # print('Cache impulse response data')
    # noise_cache = cache_noise_data(noise_paths)

    if args.memory_growth:
        physical_devices = tf.config.list_physical_devices("GPU")
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Create all the paths to the mv_set/version folders we want to test
    if settings['mv_set'] is None or len(settings['mv_set']) == 0:
        logger.error('Sample set not specified!')
        sys.exit(1)
    else:
        cand_sets = [dirname for dirname in settings['mv_set'].split(',')]
        mv_sets = []

        for cset in cand_sets:
            # Check is the directory exists
            if not os.path.exists(cset):
                logger.warning(f'Path does not exist! {cset}!')
                continue
            # Check if it directly contains wave samples            
            filenames = glob.glob(cset + '/*.wav')
            if len(filenames) > 0:
                mv_sets.append(cset)
            else:
                for subset in glob.glob(os.path.join(cset, '**/[sm]v'), recursive=True):
                    filenames = glob.glob(os.path.join(subset, '*.wav'))
                    if len(filenames) > 0:
                        mv_sets.append(os.path.join(subset))
    
    mv_sets = sorted(mv_sets)
    logger.info(f'Found speech sets to test: {len(mv_sets)}')
    for mv_set in mv_sets:
        logger.debug(f'  {mv_set}')
    assert len(mv_sets) > 0, "No valid speech sets found!"

    # Create the csv file with the similarity scores for each master voice, i.e., each master voice is compared with all the users enrolled templates
    combs = list(itertools.product(map(str, settings['net'].split(',')), map(str, settings['policy'].split(',')), map(str, settings['level'].split(',')), map(str, settings['playback'].split(','))))
    
    for net, policy, level, playback in combs:
        # Build and load pre-trained weights of a sv
        sv = verifier.get_model(net)
        sv.build(classes=0, mode='test')
        sv.load()
        sv.calibrate_thresholds()
        sv.infer()
        sv.setup_playback(settings['noise_dir'], settings['impulse_flags'])

        # Create the test gallery
        test_gallery = Dataset(settings['pop'])
        test_gallery.precomputed_embeddings(sv)
        
        if(int(playback)==1):
            iterations = settings['n_play_reps']
        else: 
            iterations = 1

        for mv_set in mv_sets:

            population_label = settings['pop'].split('/')[-1][:-4]
            fname = f'{population_label}-{net}-{policy}-{level}-{playback}.npz'.replace('/', '_')
            filename_stats = os.path.join(mv_set, fname)

            if(settings['train_analysis']):
                filename_stats =  os.path.join('./data/', fname)

            if os.path.exists(filename_stats):
                logger.warning(f'File exists ({filename_stats}) - skipping...')
                continue

            sims, imps, gnds = [], [], []

            # Repeat the test n times, each time with different playback environment
            logger.info('Testing[{}x]: net={} policy={} level={} samples={}:'.format(iterations, net, policy, level, mv_set))
            for _ in range(iterations):
                
                if(settings['train_analysis']):
                    filenames = glob.glob(settings['mv_set'] + '/*/*/*.wav')
                else: 
                    filenames = [os.path.join(mv_set, file) for file in os.listdir(mv_set) if file.endswith('.wav')]                

                # logger.info('retrieve master voice filenames {}'.format(len(filenames)))

                embeddings = sv.predict(np.array(filenames), playback=int(playback))
                # logger.info('compute master voice embeddings {}'.format(embeddings.shape))
                # logger.info('testing error rates...')

                sims_temp, imps_temp, gnds_temp = sv.test_error_rates(embeddings, test_gallery, policy=policy, level=level, playback=int(playback))

                sims.append(sims_temp)
                imps.append(imps_temp)
                gnds.append(gnds_temp)

            sims = np.array(sims)

            # Aggregate n simulated playbacks
            sims = np.mean(sims, axis=0)
            imps = np.mean(imps, axis=0)
            gnds = np.mean(gnds, axis=0)

            results = {'sims': sims, 'imps': imps, 'gnds': gnds}

            # Display current error rates
            imp_rates = results['imps'].sum(axis=1) / len(np.unique(test_gallery.user_ids))

            logger.info(f'SV using thresholds {sv._thresholds}')
            logger.warning(f'Impersonation rates [{policy}-{level}]: {100 * np.mean(imp_rates):.1f}%')
            logger.debug(f'Gender breakown [m,f]: {np.mean(100 * results["gnds"], 0).round(2)}')

            # Save results
            logger.info('saving stats to {}'.format(filename_stats))
            np.savez(filename_stats, results=results, protocol=4)

if __name__ == '__main__':
    main()

