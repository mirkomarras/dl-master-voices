#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import argparse
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'

def main():

    parser = argparse.ArgumentParser(description='Tensorflow master voice trial pairs')

    # Parameters for verifier
    parser.add_argument('--mv_set', dest='mv_set', default='', type=str, action='store', help='Folder associated to the target master voice set')
    parser.add_argument('--mv_trials', dest='mv_trials', default='./data/vs_mv_pairs/trial_pairs_vox2_mv.csv', type=str, action='store', help='Trials base path for master voice analysis waveforms')

    args = parser.parse_args()

    print('Parameters summary')
    print('>', 'Master voice set: {}'.format(args.mv_set))
    print('>', 'Master voice trials: {}'.format(args.mv_trials))

    for mv_set in ([args.mv_set] if args.mv_set else [os.path.join(mv_set, version) for mv_set in os.listdir('./data/vs_mv_data') for version in os.listdir(os.path.join('./data/vs_mv_data', mv_set))]):
        for mv_file in os.listdir(os.path.join('./data/vs_mv_data', mv_set)):
            if mv_file.endswith('.wav'):
                print('> processing set', mv_set, 'File', mv_file)
                df = pd.read_csv(args.mv_trials, names=['label', 'path1', 'gender'])
                df['path2'] = os.path.join('vs_mv_data', mv_set, mv_file)

                # Create save path
                if not os.path.exists(os.path.join('data', 'vs_mv_pairs', 'mv', mv_set)):
                    os.makedirs(os.path.join('data', 'vs_mv_pairs', 'mv', mv_set))

                df[['label', 'path1', 'path2', 'gender']].to_csv(os.path.join('data', 'vs_mv_pairs', 'mv', mv_set, mv_file.replace('.wav', '.csv')), index=False, header=False)
                print('> saved', mv_file, 'scores in', os.path.join('data', 'vs_mv_pairs', 'mv', mv_set, mv_file.replace('.wav', '.csv')))

if __name__ == '__main__':
    main()