#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import random
import csv
import os
from pathlib import Path

from helpers.audio import decode_audio

def get_mv_analysis_users(mv_analysis_path, type='all'):
    """
    Function to load the list of users related to master voice analysis
    :param mv_analysis_path:    File path to master voice analysis metadata
    :param type:                Setup for which master-voice-used users will be retrieved ['train', 'test', 'all']
    :return:                    List of users
    """

    output_users = []

    mv_analysis_data = np.load(mv_analysis_path)

    # Find the right segment
    pid = None
    items = mv_analysis_data['x_train'][0].split('/')
    for i, item in enumerate(items):
        if items[i].startswith('id'):
            pid = i

    assert pid is not None, f"None of the elements in file path is recognized as user iD: {items}"

    if type in ['all', 'train']:
        print('Load train user ids for mv analysis')
        train_users = list(np.unique([path.split('/')[pid] for path in mv_analysis_data['x_train']]))
        assert train_users[0].startswith('id'), f"The user IDs should start with 'id' - currently seeing {train_users[0]}"
        output_users = output_users + train_users
        print('>', 'found mv analysis train users', len(train_users))

    if type in ['all', 'test']:
        print('Load test user ids for mv analysis')
        test_users = list(np.unique([path.split('/')[pid] for path in mv_analysis_data['x_test']]))
        assert test_users[0].startswith('id'), f"The user IDs should start with 'id' - currently seeing {train_users[0]}"
        output_users = output_users + test_users
        print('>', 'found mv analysis test users', len(test_users))

    return output_users


def load_data_set(audio_dir, mv_user_ids, include=False, n_samples=None):
    """
    Function to load an audio data with format {user_id}/{video_id}/xyz.wav
    :param audio_dir:       List of base paths to datasets
    :param mv_user_ids:     List of user ids that will be excluded from loading
    :param include:         Flag to exclude master-voice-used users - with include=False, master-voice-used users will be excluded
    :return:                (List of audio file paths, List of user labels)
    """

    x = []
    y = []

    print('Load data sets')
    user_count = 0

    for source_dir in audio_dir:

        print(f'> loading data from {source_dir} (sample user id: {mv_user_ids[0]})', )

        for user_id, user_dir in enumerate(os.listdir(source_dir)):

            if (include and user_dir in mv_user_ids) or (not include and user_dir not in mv_user_ids):

                audio_files = [str(x) for x in Path(os.path.join(source_dir, user_dir)).glob('**/*.wav')]
                if n_samples is not None:
                    audio_files = audio_files[:n_samples]

                x.extend(audio_files)
                y.extend([user_count for x in audio_files])

                user_count += 1

    print('>', 'loaded', len(x), 'audio files from', len(np.unique(y)), 'users totally')

    return x, y


def filter_by_gender(paths, labels, meta_file, gender='neutral'):
    """
    Function to filter audio files based on the gender of the speaking user
    :param paths:       List of audio file paths
    :param labels:      List of users' labels
    :param meta_file:   Path to the file with gender information
    :param gender:      Targeted gender to keep
    :return:            List of paths from users with the targeted gender
    """

    print('Filter data sets by gender', gender)
    data_set_df = pd.read_csv(meta_file, delimiter=' ')
    gender_map = {k:v for k, v in zip(data_set_df['id'].values, data_set_df['gender'].values)}

    filtered_paths = []
    filtered_labels = []

    if gender == 'male' or gender == 'female':

        for path, label in zip(paths, labels):
            if gender_map[path.split(os.path.sep)[-3]] == gender[0]:
                filtered_paths.append(path)
                filtered_labels.append(label)

        print('>', 'filtered', len(filtered_paths), 'audio files from', len(np.unique(filtered_labels)), 'users')

        return filtered_paths, filtered_labels

    print('>', 'remaining', len(paths), 'audio files from', len(np.unique(labels)), 'users')

    return paths, labels


def load_test_data_from_file(base_path, trials_path, n_templates=1, n_pairs=10, sample_rate=16000, n_seconds=3, print_interval=100):
    """
    Function lo load raw audio samples for testing
    :param base_path:       Base path to the dataset samples
    :param trials_path:     Path to the list of trial pairs
    :param n_pairs:         Number of pairs to be loaded
    :param sample_rate:     Sample rate of the audio files to be processed
    :param n_seconds:       Max number of seconds of an audio sample to be processed
    :return:                (list of audio samples, list of audio samples), labels
    """

    print('Loading testing data from file', trials_path, 'with template', n_templates)

    pairs = pd.read_csv(trials_path, names=['target','path_1','path_2'], delimiter=' ')
    n_real_pairs = n_pairs if n_pairs > 0 else len(pairs['target'])

    y = pairs['target'].values[:n_real_pairs]
    x1 = []
    x2 = []

    for i, (path_1, path_2) in enumerate(zip(pairs['path_1'].values[:n_real_pairs], pairs['path_2'].values[:n_real_pairs])):
        if (i+1) % print_interval == 0:
            print('\r> pair %5.0f / %5.0f' % (i+1, len(y)), end='')

        x1.append(decode_audio(os.path.join(base_path, path_1), sample_rate=sample_rate).reshape((1, -1, 1)))
        x2.append([decode_audio(os.path.join(base_path, path), sample_rate=sample_rate).reshape((1, -1, 1)) for path in (path_2 if isinstance(path_2, list) else [path_2])])

    print('\n>', 'found', len(x1), 'pairs')

    return (x1, x2), y


def create_template_trials(base_path, trials_path, n_templates=1, n_pos_pairs=10, n_neg_pairs=10):
    users = os.listdir(base_path)
    print('> creating pairs on', len(users), 'users')

    groups_audios = {}
    counter = 0
    for i, user in enumerate(users):
        print('\r> grouping videos for user', i+1, '/', len(users), end='')
        if not user in groups_audios:
            groups_audios[user] = []
        for video in os.listdir(os.path.join(base_path, user)):
            for utterance in os.listdir(os.path.join(base_path,user,video)):
                groups_audios[user].append(os.path.join(user,video,utterance))
                counter += 1
    print('\n> loaded', counter, 'samples')

    with open(trials_path, mode='w') as result_file:
        result_writer = csv.writer(result_file, delimiter=',')
        print('> expected', len(users) * (n_pos_pairs + n_neg_pairs), 'pairs')
        counter_pairs = 1
        for index_user, curr_user in enumerate(users):
            print('\r> manipulating pairs for user', index_user+1, '/', len(users), end='')
            neg_users = list(set(users) - set(np.array([curr_user])))
            for pos_index in range(n_pos_pairs):
                template_audio = np.random.choice(groups_audios[curr_user], n_templates, replace=True).tolist()
                probe_audio = random.choice(list(set(groups_audios[curr_user]) - set(template_audio)))
                result_writer.writerow([1, probe_audio, probe_audio])
                result_file.flush()
                counter_pairs += 1
            for neg_index in range(n_neg_pairs):
                template_audio = np.random.choice(groups_audios[curr_user], n_templates, replace=True).tolist()
                other_user = random.choice(neg_users)
                other_audio = random.choice(groups_audios[other_user])
                result_writer.writerow([0, other_audio, template_audio])
                result_file.flush()
                counter_pairs += 1
    print('> computed', counter_pairs-1, 'pairs')


def load_mv_data(mv_analysis_path, mv_base_path, audio_meta, sample_rate=16000, n_templates=10, type='test'):
    """
    Function to load data for master voice impersonation
    :param mv_analysis_path:    File path to master voice analysis metadata
    :param mv_base_path:        Base path of the dataset from which master-voice-used audio samples are retrieved
    :param audio_meta:          Path to the file with gender information
    :param sample_rate:         Sample rate of the audio files to be processed
    :param n_seconds:           Max number of seconds of an audio sample to be processed
    :param n_templates:         Number of audio samples per user to be loaded
    :return:                    (list of audio samples, list of labels, list of male user ids, list of female user ids)
    """
    print('Loading audio files for master voice validation')

    mv_analysis_data = np.load(mv_analysis_path)
    mv_paths = [os.path.join(mv_base_path, path) for path in mv_analysis_data['x_' + type]]
    mv_labels = mv_analysis_data['y_test']
    print('> found', len(mv_paths), 'paths from', len(np.unique(mv_labels)), 'users')

    data_set_df = pd.read_csv(audio_meta, delimiter=' ')
    gender_map = {k:v for k, v in zip(data_set_df['id'].values, data_set_df['gender'].values)}

    x_mv_test, y_mv_test, male_x_mv_test, female_x_mv_test = [], [], [], []
    samples_per_user = int(len(mv_paths) // len(np.unique(mv_labels)))

    for class_index, _ in enumerate(np.unique(mv_labels)):

        class_paths = random.sample(mv_paths[class_index*samples_per_user:(class_index+1)*samples_per_user], n_templates)

        for path in class_paths:
            # TODO ugly workaround
            path = path.replace('dev/dev/', 'dev/')
            x_mv_test.append(decode_audio(path.replace('.m4a', '.wav'), sample_rate=sample_rate).reshape((1, -1, 1)))
            y_mv_test.append(class_index)

        if gender_map[class_paths[0].split(os.path.sep)[-3]] == 'm':
            male_x_mv_test.append(class_index)
        else:
            female_x_mv_test.append(class_index)

        print('\r> loaded', (class_index+1)*n_templates, '/', len(np.unique(mv_labels))*n_templates, 'audio files', end='')

    print()

    return x_mv_test, y_mv_test, male_x_mv_test, female_x_mv_test


def load_mv_list(mv_analysis_path, mv_base_path, audio_meta, n_templates=10):
    """
    Function to load data for master voice impersonation
    :param mv_analysis_path:    File path to master voice analysis metadata
    :param mv_base_path:        Base path of the dataset from which master-voice-used audio samples are retrieved
    :param audio_meta:          Path to the file with gender information
    :param sample_rate:         Sample rate of the audio files to be processed
    :param n_seconds:           Max number of seconds of an audio sample to be processed
    :param n_templates:         Number of audio samples per user to be loaded
    :return:                    (list of audio samples, list of labels, list of male user ids, list of female user ids)
    """
    print('Loading master voice data')

    mv_analysis_data = np.load(mv_analysis_path)
    mv_paths = [os.path.join(mv_base_path, path) for path in mv_analysis_data['x_test']]
    mv_labels = mv_analysis_data['y_test']
    print('> found', len(mv_paths), 'paths from', len(np.unique(mv_labels)), 'users')

    data_set_df = pd.read_csv(audio_meta, delimiter=' ')
    gender_map = {k:v for k, v in zip(data_set_df['id'].values, data_set_df['gender'].values)}

    x_mv_test, y_mv_test, g_mv_test = [], [], []
    samples_per_user = int(len(mv_paths) // len(np.unique(mv_labels)))

    for class_index, _ in enumerate(np.unique(mv_labels)):

        class_paths = random.sample(mv_paths[class_index*samples_per_user:(class_index+1)*samples_per_user], n_templates)

        for path in class_paths:
            x_mv_test.append(path.replace('.m4a', '.wav'))
            y_mv_test.append(class_index)
            g_mv_test.append(gender_map[class_paths[0].split(os.path.sep)[-3]])

        print('\r> loaded', (class_index+1)*n_templates, '/', len(np.unique(mv_labels))*n_templates, 'audio files', end='')

    print()

    return x_mv_test, y_mv_test, g_mv_test
