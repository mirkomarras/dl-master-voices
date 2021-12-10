#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import itertools
import os
import glob 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from loguru import logger
import numpy as np
import argparse

from helpers.dataset import Dataset
from models import verifier

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main():
    parser = argparse.ArgumentParser(description='Tensorflow master voice trial pairs generation and testing')

    parser.add_argument('--net', dest='net', default='vggvox/v000,resnet50/v000,thin_resnet/v000,resnet34/v000', type=str, action='store', help='Speaker model, e.g., vggvox/v003') #vggvox/v004,resnet50/v004,thin_resnet/v002,resnet34/v002,xvector/v003

    parser.add_argument('--playback', dest='playback', default='0,1', type=str, action='store', help='Playback and recording in master voice test condition: 0 no, 1 yes')
    parser.add_argument('--noise_dir', dest='noise_dir', default='data/vs_noise_data', type=str, action='store', help='Noise directory')

    parser.add_argument('--mv_set', dest='mv_set', default='vggvox_v000_pgd_spec_m/v028/sv,vggvox_v000_pgd_spec_m/v028/mv', action='store', help='Directory with MV data')
    parser.add_argument('--pop', dest='pop', default='data/vs_mv_pairs/mv_test_population_is2019_100u_10s.csv', type=str, action='store', help='Path to the filename-user_id pairs for mv training')
    parser.add_argument('--policy', dest='policy', default='any,avg', type=str, action='store', help='Policy of verification, eigher any or avg')
    parser.add_argument('--level', dest='level', default='eer,far1', type=str, action='store', help='Levelof security, either eer or far1')
    parser.add_argument('--n_random', dest='n_random', default=20, type=int, action='store', help='Number of randomized playback settings to run the test for')
    parser.add_argument('-impulse_flags','--impulse_flags', nargs='+', help='Impulse Flags for controlling playback', required=False)
    parser.add_argument('--roll', dest='roll', default=None, type=int, action='store', help='Length to rolling the audio sample')
    parser.add_argument('--train_analysis', dest='train_analysis', default=False, type=bool, action='store', help='Picking filenames for train analysis')
    
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
        mv_sets = [os.path.join('./', 'data', 'vs_mv_data', mv_set, version) for mv_set in os.listdir(os.path.join('.', 'data', 'vs_mv_data')) for version in os.listdir(os.path.join('.', 'data', 'vs_mv_data', mv_set))]
    else:
        mv_sets = [os.path.join('./', 'data', 'vs_mv_data', mv_set) for mv_set in settings['mv_set'].split(',')]

    logger.info(mv_sets)

    # Create the csv file with the similarity scores for each master voice, i.e., each master voice is compared with all the users enrolled templates
    combs = list(itertools.product(map(str, settings['net'].split(',')), map(str, settings['policy'].split(',')), map(str, settings['level'].split(',')), map(str, settings['playback'].split(','))))
    for net, policy, level, playback in combs:
        # Build and load pre-trained weights of a sv
        sv = verifier.get_model(net)
        sv.build(classes=0, mode='test')
        sv.load()
        sv.calibrate_thresholds()
        sv.infer()

        # Create the test gallery
        test_gallery = Dataset(settings['pop'])
        test_gallery.precomputed_embeddings(sv)
        # print("SEEMS LIKE I AM PAST THIS AS WELL")
        if(int(playback)==1):
            iterations = settings['n_random']
        else: 
            iterations = 1


        
        for mv_set in mv_sets:
            sims, imps, gnds = [], [], []

            for _ in range(iterations):

                logger.info('testing setup net={} policy={} level={} mv_set={}:'.format(net, policy, level, mv_set))
                
                # print(filenames)
                if(settings['train_analysis']):


                    filenames = glob.glob(settings['mv_set'] + '/*/*/*.wav')
                else: 
                    filenames = [os.path.join(mv_set, file) for file in os.listdir(mv_set) if file.endswith('.wav')]
                
                # print("Filenames", filenames)

                logger.info('retrieve master voice filenames {}'.format(len(filenames)))

                embeddings = sv.predict(np.array(filenames), playback=int(playback))
                logger.info('compute master voice embeddings {}'.format(embeddings.shape))

                logger.info('testing error rates...')


                sims_temp, imps_temp, gnds_temp = sv.test_error_rates(embeddings, test_gallery, policy=policy, level=level, playback=int(playback))

                sims.append(sims_temp)
                imps.append(imps_temp)
                gnds.append(gnds_temp)

            sims = np.array(sims)

            sims = np.mean(sims, axis=0)
            imps = np.mean(imps, axis=0)
            gnds = np.mean(gnds, axis=0)
            

            results = {'sims': sims, 'imps': imps, 'gnds': gnds}
            
            filename_stats = os.path.join(mv_set, settings['pop'].split('/')[-1][:-4] + '-' + net.replace('/', '_') + '-' + str(policy) + '-' + str(level) + '-' + str(playback)+ '.npz')
            if(settings['train_analysis']):
                filename_stats =  os.path.join('./data/', settings['pop'].split('/')[-1][:-4] + '-' + net.replace('/', '_') + '-' + str(policy) + '-' + str(level) + '-' + str(playback)+ '.npz')
            logger.info('saving stats to {}'.format(filename_stats))
            np.savez(filename_stats, results=results, protocol=4)

if __name__ == '__main__':
    main()

