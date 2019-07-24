from helpers.coreutils import *
import numpy as np
import os

def getData(source_vox1, source_vox2):
    paths = []
    labels = []
    max_lenghts = []
    counter = 0
    excluded_users = get_excluded_users()
    for source in [source_vox1, source_vox2]:
        source = os.path.join(source, 'dev')
        max_lenghts.append(len(list(os.listdir(source))))
        print('Start loading data from', source)
        for index_user_folder, user_folder in enumerate(os.listdir(source)):
            if not user_folder in excluded_users:
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
    train_users = np.unique([p.split('/')[1] for p in train_paths])
    print('Found mv train users', len(train_users))

    test_paths = load_obj('./data/vox2_mv/test_vox2_abspaths_1000_users')
    test_users = np.unique([p.split('/')[1] for p in test_paths])
    print('Found mv test users', len(test_users))

    excluded_users = list(train_users) + list(test_users)
    return excluded_users