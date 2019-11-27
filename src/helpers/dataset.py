#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import random
import os

from src.helpers.audio import decode_audio

def get_mv_analysis_users(mv_analysis_path, type='all'):
    output_users = []
    mv_analysis_data = np.load(mv_analysis_path)
    if type in ['all', 'train']:
        print('Load train user ids for mv analysis')
        train_users = list(np.unique([path.split('/')[1] for path in mv_analysis_data['x_train']]))
        output_users = output_users + train_users
        print('>', 'found mv analysis train users', len(train_users))
    if type in ['all', 'test']:
        print('Load test user ids for mv analysis')
        test_users = list(np.unique([path.split('/')[1] for path in mv_analysis_data['x_test']]))
        output_users = output_users + test_users
        print('>', 'found mv analysis test users', len(test_users))
    return output_users

def load_data_set(audio_dir, mv_user_ids, include=False):
    print('Load data sets')

    x = []
    y = []

    user_count = 0
    for source_dir in audio_dir:
        for user_id, user_dir in enumerate(os.listdir(os.path.join(source_dir))):
            if (include and user_dir in mv_user_ids) or (not include and user_dir not in mv_user_ids):
                for video_id, video_dir in enumerate(os.listdir(os.path.join(source_dir, user_dir))):
                    for audio_id, audio_file in enumerate(os.listdir(os.path.join(source_dir, user_dir, video_dir))):
                        x.append(os.path.join(source_dir, user_dir, video_dir, audio_file))
                        y.append(user_count)
                user_count += 1

    print('>', 'loaded', len(x), 'audio files from', len(np.unique(y)), 'users')

    return x, y

def filter_by_gender(paths, labels, meta_file, gender='neutral'):
    print('Filter data sets by gender', gender)
    data_set_df = pd.read_csv(meta_file, delimiter=' ')
    gender_map = {k:v for k, v in zip(data_set_df['id'].values, data_set_df['gender'].values)}

    filtered_paths = []
    filtered_labels = []

    if gender is not 'neutral':
        for path, label in zip(paths, labels):
            if gender_map[path.split(os.path.sep)[-3]] == gender[0]:
                filtered_paths.append(path)
                filtered_labels.append(label)
        print('>', 'remaining', len(filtered_paths), 'audio files from', len(np.unique(filtered_labels)), 'users')
        return filtered_paths, filtered_labels

    print('>', 'remaining', len(paths), 'audio files from', len(np.unique(labels)), 'users')
    return paths, labels

def load_data_raw(base_path, trials_path, n_pairs=10, sample_rate=16000, n_seconds=3, print_interval=100):
    pairs = pd.read_csv(trials_path, names=['target','path_1','path_2'], delimiter=' ')
    n_real_pairs = n_pairs if n_pairs > 0 else len(pairs['target'])
    y = pairs['target'].values[:n_real_pairs]
    x1 = np.zeros((len(y), sample_rate*n_seconds, 1))
    x2 = np.zeros((len(y), sample_rate*n_seconds, 1))
    for i, (path_1, path_2) in enumerate(zip(pairs['path_1'].values[:n_real_pairs], pairs['path_2'].values[:n_real_pairs])):
        if (i+1) % print_interval == 0:
            print('\r> pair %5.0f / %5.0f' % (i+1, len(y)), end='')
        x1[i] = decode_audio(os.path.join(base_path, path_1), sample_rate=sample_rate).reshape((-1, 1))[:sample_rate*n_seconds, :]
        x2[i] = decode_audio(os.path.join(base_path, path_2), sample_rate=sample_rate).reshape((-1, 1))[:sample_rate*n_seconds, :]
    return (x1, x2), y

def load_val_data(base_path, trials_path, n_pairs=10, sample_rate=16000, n_seconds=3):
    print('Loading validation data')
    (x1_val, x2_val), y_val = load_data_raw(base_path, trials_path, n_pairs, sample_rate, n_seconds)
    print('\n>', 'found', len(x1_val), 'pairs')
    return (x1_val, x2_val), y_val

def load_test_data(base_path, trials_path, n_pairs=10, sample_rate=16000, n_seconds=3):
    print('Loading testing data')
    (x1_test, x2_test), y_test = load_data_raw(base_path, trials_path, n_pairs, sample_rate, n_seconds)
    print('\n>', 'found', len(x1_test), 'pairs')
    return (x1_test, x2_test), y_test

def load_mv_data(mv_analysis_path, mv_base_path, audio_meta, sample_rate=16000, n_seconds=3, n_templates=10):
    print('Loading master voice data')

    mv_analysis_data = np.load(mv_analysis_path)
    mv_paths = [os.path.join(mv_base_path, path) for path in mv_analysis_data['x_test']]
    mv_labels = mv_analysis_data['y_test']
    print('> found', len(mv_paths), 'paths from', len(np.unique(mv_labels)), 'users')

    data_set_df = pd.read_csv(audio_meta, delimiter=' ')
    gender_map = {k:v for k, v in zip(data_set_df['id'].values, data_set_df['gender'].values)}

    x_mv_test, y_mv_test, male_x_mv_test, female_x_mv_test = [], [], [], []
    samples_per_user = int(len(mv_paths) // len(np.unique(mv_labels)))

    for class_index, _ in enumerate(np.unique(mv_labels)):

        class_paths = random.sample(mv_paths[class_index*samples_per_user:(class_index+1)*samples_per_user], n_templates)

        for path in class_paths:
            x_mv_test.append(decode_audio(path, sample_rate=sample_rate).reshape((-1, 1))[:sample_rate*n_seconds, :])
            y_mv_test.append(class_index)

        if gender_map[class_paths[0].split(os.path.sep)[-3]] == 'm':
            male_x_mv_test.append(class_index)
        else:
            female_x_mv_test.append(class_index)

        print('\r> loaded', (class_index+1)*n_templates, '/', len(np.unique(mv_labels))*n_templates, 'audio files', end='')

    print()

    return x_mv_test, y_mv_test, male_x_mv_test, female_x_mv_test