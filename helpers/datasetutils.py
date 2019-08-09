from helpers.coreutils import *
import pandas as pd
import numpy as np
import os

def getData(sources, gender=''):
    paths = []
    labels = []
    max_lenghts = []
    counter = 0
    # TODO Fix this!
    excluded_users = []
    # excluded_users = get_excluded_users()

    valid_users = []
    for source in sources:
        meta_df = pd.read_csv(os.path.join(source, 'meta.csv'), delimiter='\t')
        # TODO This should be limited to dev section - skip for now
        # meta_df = meta_df[meta_df['Set']=='dev']

        if any(x not in meta_df for x in ['ID', 'Gender']):
            raise ValueError('Invalid meta.msv for {}'.format(source))

        if gender in ['m', 'f']:
            valid_users += meta_df[meta_df['Gender'] == gender]['ID'].values.tolist()
        else:
            valid_users += meta_df['ID'].values.tolist()

    print('Found total users', len(valid_users))
    print('Found excluded users', len(excluded_users))

    for source in sources:
        source = os.path.join(source, 'dev')
        max_lenghts = len(list(os.listdir(source)))
        print('Start loading data from', source)
        for index_user_folder, user_folder in enumerate(os.listdir(source)):
            if (not user_folder in excluded_users) and (user_folder in valid_users):
                print('\rLoading utterances for user {}/{}'.format(counter+1, max_lenghts), end='')
                user_folder_path = os.path.join(source, user_folder)
                for index_video_folder, video_folder in enumerate(os.listdir(user_folder_path)):
                    video_folder_path = os.path.join(source, user_folder, video_folder)
                    for index_audio_folder, audio_folder in enumerate(os.listdir(video_folder_path)):
                        paths.append(os.path.join(source, user_folder, video_folder, audio_folder))
                        labels.append(counter)
                counter += 1
        print()

    print('Found', len(paths), 'from', len(np.unique(labels)), 'users')

    return {"paths": paths, "labels": labels}

def get_excluded_users():
    train_paths = load_obj('./data/vox2_mv/train_vox2_abspaths_1000_users')
    train_users = np.unique([p.split('/')[5] for p in train_paths])
    print('Found mv train users', len(train_users))

    test_paths = load_obj('./data/vox2_mv/test_vox2_abspaths_1000_users')
    test_users = np.unique([p.split('/')[4] for p in test_paths])
    print('Found mv test users', len(test_users))

    excluded_users = list(train_users) + list(test_users)
    return excluded_users
