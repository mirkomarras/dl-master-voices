

import numpy as np
import os
import random
from pathlib import Path


# Functions to select and enroll the speakers:
# The resulting filenames will be stored in plaintext files (filename, user ID)
# E.g., for the paper:
#   data/vs_mv_data/mv_train_population_tifs2021_1000u_100s.txt
#   data/vs_mv_data/mv_test_population_tifs2021_1000u_100s.txt
# E.g., for local debugging:
#   data/vs_mv_data/mv_test_population_debug_100u_10s.txt

def generate_mv_training_samples():
    """ list: filename, user ID """
    pass

def generate_mv_testing_samples():
    """ list: filename, user ID """
    pass


def generate_enrolled_samples(npz_file='data/vs_mv_pairs/data_mv_vox2_debug.npz', dirname='data/voxceleb2/dev', n_train=20, n_test=10, n_split=(20, 20)):
    """
    Generates 
    """

    users = os.listdir(dirname)
    random.shuffle(users)

    # x_train -> ignored
    # y_train -> unique -> sampling x_train from scratch

    # x_test - filename with speech sample
    # y_test - speaker identity ID
    data = {k: [] for k in ('x_train', 'y_train', 'x_test', 'y_test')}

    print(f'Looking for files in {dirname}')

    print('# Training population')
    for u in users[:n_split[0]]:
        files = [str(f).replace(dirname + '/', '') for f in Path(os.path.join(dirname, u)).glob('**/*.m4a')][:n_train]
        print(u, len(files), 'files')
        data['x_train'].extend(files)
        data['y_train'].extend([int(u[2:]) for f in files])

    print('# Testing population')
    for u in users[n_split[0]:sum(n_split)]:
        files = [str(f).replace(dirname + '/', '') for f in Path(os.path.join(dirname, u)).glob('**/*.m4a')][:n_test]
        print(u, len(files), 'files')
        data['x_test'].extend(files)
        data['y_test'].extend([int(u[2:]) for f in files])

    print('# Data')
    pprint(data)

    print([len(x) for k, x in data.items()])

    print(f'Writing to {npz_file}')
    # np.savez(npz_file, **data)

    return data


# Functions to work with cached speaker embeddings
# We can store the speaker embeddings for the test population for all SV systems 
#   data/vs_mv_data/mv_test_population_debug_100u_10s/embeddings/xvector_v004.npz
#   data/vs_mv_data/mv_test_population_debug_100u_10s/embeddings/vggvox_v004.npz

def precompute_embedding(speech_filenames):
    pass

def load_embeddings(speech_filenames):
    pass

# Storing similarity scores:
#   data/vs_mv_data/mv_test_population_debug_100u_10s/similarity/xvector_v004.npz

# Gallery_samples
# x_test = [P01_001.wav, P01_002.wav, ...., P10_001.wav, P10_010.wav]

#  MV sample \ Gallery sample |  P01_001.wav | P01_002.wav | .... | P10_001.wav | P10_010.wav
#  MV 001                     |         0.90 | 
#  MV 002                     |


def initialize_array(n_test, n_enrolled):
    return np.zeros((n_test, n_enrolled))

# Storing results / figures
#   data/vs_mv_data/mv_test_population_debug_100u_10s/results/xvector_v004.*



# Batch processing functions

