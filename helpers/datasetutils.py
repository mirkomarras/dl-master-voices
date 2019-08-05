from helpers.coreutils import *
import pandas as pd
import numpy as np
import os

def getData(source_vox1, source_vox2, gender=''):
    paths = []
    labels = []
    max_lenghts = []
    counter = 0
    excluded_users = get_excluded_users()

    vox1_meta = pd.read_csv('./data/vox1_meta/meta_vox1.csv', delimiter='\t')
    vox1_meta = vox1_meta[vox1_meta['Set']=='dev']
    vox2_meta = pd.read_csv('./data/vox2_meta/meta_vox2.csv', header=None, names=['VoxCeleb2 ID','VGGFace2 ID', 'Gender', 'Set'])
    vox2_meta = vox2_meta[vox2_meta['Set']=='dev ']
    valid_users = []
    if gender == '':
        print('Found Vox1 users', len(list(vox1_meta['VoxCeleb1 ID'].values)))
        valid_users += list(vox1_meta['VoxCeleb1 ID'].values)
        print('Found Vox2 users', len(list(vox2_meta['VoxCeleb2 ID'].values)))
        valid_users += list(vox2_meta['VoxCeleb2 ID'].values)
    if gender == 'm':
        print('Found Vox1 users', len(list(vox1_meta[vox1_meta['Gender']=='m']['VoxCeleb1 ID'].values)))
        valid_users += list(vox1_meta[vox1_meta['Gender']=='m']['VoxCeleb1 ID'].values)
        print('Found Vox2 users', len(list(vox2_meta[vox2_meta['Gender']=='m']['VoxCeleb2 ID'].values)))
        valid_users += list(vox2_meta[vox2_meta['Gender']=='m']['VoxCeleb2 ID'].values)
    if gender == 'f':
        print('Found Vox1 users', len(list(vox1_meta[vox1_meta['Gender']=='f']['VoxCeleb1 ID'].values)))
        valid_users += list(vox1_meta[vox1_meta['Gender']=='f']['VoxCeleb1 ID'].values)
        print('Found Vox2 users', len(list(vox2_meta[vox2_meta['Gender']=='f']['VoxCeleb2 ID'].values)))
        valid_users += list(vox2_meta[vox2_meta['Gender ']=='f']['VoxCeleb2 ID'].values)

    print('Found total users', len(valid_users))
    print('Found excluded users', len(excluded_users))

    for source in [source_vox1, source_vox2]:
        source = os.path.join(source, 'dev')
        max_lenghts.append(len(list(os.listdir(source))))
        print('Start loading data from', source)
        for index_user_folder, user_folder in enumerate(os.listdir(source)):
            if (not user_folder in excluded_users) and (user_folder in valid_users):
                print('\rLoading utterances from user', counter+1, '/', np.sum(max_lenghts), end='')
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