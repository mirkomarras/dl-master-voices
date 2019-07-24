from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pickle

def save_obj(obj, name):
    print('Saving object', name)
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    print('Loading object', name)
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def save_npy(obj, name):
    print('Saving numpy', name)
    np.save(name + '.npy', obj)

def load_npy(name):
    print('Loading numpy', name)
    return np.load(name + '.npy')