#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.metrics import roc_curve
import tensorflow as tf
import soundfile as sf
import pandas as pd
import numpy as np
import argparse
import librosa
import pickle
import os

from helpers.dataset import get_mv_analysis_users, load_data_set, filter_by_gender
from helpers.audio import get_tf_spectrum, get_tf_filterbanks, decode_audio
from models.verifier.xvector import normalize_with_moments
from models.verifier.model import VladPooling

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'

def main():

    parser = argparse.ArgumentParser(description='Tensorflow master voice trial pairs')

    # Parameters for verifier
    parser.add_argument('--mv_base_path', dest='mv_base_path', default='/beegfs/mm11333/data/voxceleb2/dev', type=str, action='store', help='Trials base path for master voice analysis waveforms')
    parser.add_argument('--mv_meta', dest='mv_meta', default='./data/ad_voxceleb12/vox2_mv_data.npz', type=str, action='store', help='Numpy data for master voice analysis metadata')
    parser.add_argument('--audio_meta', dest='audio_meta', default='./data/ad_voxceleb12/vox12_meta_data.csv', type=str, action='store', help='CSV file with id-gender metadata')
    parser.add_argument('--n_comparisons', dest='n_comparisons', type=int, default=37720, action='store', help='Number of comparisons per model')
    parser.add_argument('--n_templates', dest='n_templates', type=int, default=10, action='store', help='Enrolment set size')
    parser.add_argument('--policy', dest='policy', default='single', type=str, action='store', help='CSV file with id-gender metadata')

    args = parser.parse_args()

    print('Parameters summary')
    print('>', 'Master voice base path: {}'.format(args.mv_base_path))
    print('>', 'Master voice meta path: {}'.format(args.mv_meta))
    print('>', 'Audio meta path: {}'.format(args.audio_meta))
    print('>', 'Number of comparisons: {}'.format(args.n_comparisons))
    print('>', 'Number of samples per template: {}'.format(args.n_templates))
    print('>', 'Policy: {}'.format(args.policy))

    print('Loading utterances')
    mv_user_ids = get_mv_analysis_users(args.mv_meta, type='train')
    x_train, y_train = load_data_set([args.mv_base_path], mv_user_ids, include=True)
    x_train_male, y_train_male = filter_by_gender(x_train, y_train, args.audio_meta, 'male')
    x_train_female, y_train_female = filter_by_gender(x_train, y_train, args.audio_meta, 'female')

    dict_gen = {'male': [], 'female': []}
    dict_all = {}

    for (x, y) in zip(x_train_female, y_train_female):
        if y not in dict_all:
            dict_all[y] = []
            dict_gen[y] = 'female'
        dict_all[y].append(x)

    for (x, y) in zip(x_train_male, y_train_male):
        if y not in dict_all:
            dict_all[y] = []
            dict_gen[y] = 'male'
        dict_all[y].append(x)

    print('>', len(list(dict_all.keys())), list(dict_gen.values()).count('female'), list(dict_gen.values()).count('male'))

    identical = []
    end_paths_1 = []
    end_paths_2 = []

    if args.policy == 'single':
        for index in range(args.n_comparisons):
            user_1 = np.random.choice(list(dict_all.keys()))
            user_2 = np.random.choice(list(set(list(dict_all.keys())) - set(list([user_1]))))

            path_1_1 = np.random.choice(dict_all[user_1])
            path_1_2 = np.random.choice(list(set(dict_all[user_1]) - set(list([path_1_1]))))
            path_2 = np.random.choice(dict_all[user_2])

            identical.append(1)
            end_paths_1.append(path_1_1)
            end_paths_2.append(path_1_2)

            identical.append(0)
            end_paths_1.append(path_1_1)
            end_paths_2.append(path_2)

            print('\r>', index+1, 'of', args.n_comparisons, end='')

        df = pd.DataFrame(list(zip(identical, end_paths_1, end_paths_2)), columns=['target', 'path1', 'path2'])
        print('\nBefore', len(df.index))
        df = df.drop_duplicates(keep='first')
        df = df.head(args.n_comparisons)
        print('After', len(df.index))
        df['path1'] = df['path1'].apply(lambda x: x.replace('/beegfs/mm11333/data/voxceleb2/dev/', ''))
        df['path2'] = df['path2'].apply(lambda x: x.replace('/beegfs/mm11333/data/voxceleb2/dev/', ''))
        df.to_csv(os.path.join('/beegfs/mm11333/dl-master-voices/data/vs_mv_pairs', 'trial_pairs_vox2_test.csv'), index=False, header=False, sep=' ')

    elif args.policy == 'multiple':
        for index in range(args.n_comparisons):
            user_1 = np.random.choice(list(dict_all.keys()))
            user_2 = np.random.choice(list(set(list(dict_all.keys())) - set(list([user_1]))))

            path_1_1 = np.random.choice(dict_all[user_1])
            path_1_2 = np.random.choice(list(set(dict_all[user_1]) - set(list([path_1_1]))), args.n_templates)
            path_2 = np.random.choice(dict_all[user_2], args.n_templates)

            identical.append(1)
            end_paths_1.append(path_1_1)
            end_paths_2.append(path_1_2)

            identical.append(0)
            end_paths_1.append(path_1_1)
            end_paths_2.append(path_2)

            print('\r>', index+1, 'of', args.n_comparisons, end='')

        df = pd.DataFrame(list(zip(identical, end_paths_1, end_paths_2)), columns=['target', 'path1', 'path2'])
        print('\nBefore', len(df.index))
        df = df.drop_duplicates(keep='first')
        df = df.head(args.n_comparisons)
        print('After', len(df.index))
        df['path1'] = df['path1'].apply(lambda x: x.replace('/beegfs/mm11333/data/voxceleb2/dev/', ''))
        df['path2'] = df['path2'].apply(lambda x: x.replace('/beegfs/mm11333/data/voxceleb2/dev/', ''))
        df.to_csv(os.path.join('/beegfs/mm11333/dl-master-voices/data/vs_mv_pairs', 'trial_pairs_vox2_test.csv'), index=False, header=False, sep=' ')

    else:
        raise NotImplementedError('Policy not implemented!')

if __name__ == '__main__':
    main()