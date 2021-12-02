import numpy as np
import os
import random
from pathlib import Path
from pprint import pprint

npz_file = 'data/vs_mv_pairs/data_mv_vox2_debug.npz'
dirname = 'data/voxceleb2/dev'
n_train = 20
n_test = 10
n_split = (20, 20)

users = os.listdir(dirname)
random.shuffle(users)

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
np.savez(npz_file, **data)