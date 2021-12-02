#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 15:08:07 2021

@author: pkorus
"""
import tensorflow as tf

from models.verifier import xvector

sv = xvector.XVector(id=0)
sv.build(0)
sv.calibrate_thresholds()
sv.load()
sv.infer()

# x.model.summary()

# x.build(5205)

# from models.verifier.model import VladPooling

# %%
model_path = 'data/vs_mv_models/xvector/v000/model.h5'
# x.model.load_weights(model_path, skip_mismatch=True, by_name=True)

# x.model.load_weights(model_path)

# X = tf.keras.models.load_model(model_path, custom_objects={'VladPooling': VladPooling})

# %%

import numpy as np
import os
from helpers.dataset import Dataset


gallery = Dataset('data/vs_mv_pairs/mv_test_population_interspeech_1000u_10s.csv')
gallery.precomputed_embeddings(sv)

mv_set = 'data/vs_mv_seed/female/'

filenames = [os.path.join(mv_set, file) for file in os.listdir(mv_set) if file.endswith('.wav')]
# logger.info('retrieve master voice filenames {}'.format(len(filenames)))

embeddings = sv.predict(np.array(filenames))


sim_matrix, imp_matrix, gnd_matrix = sv.test_error_rates(embeddings, gallery, policy='avg', level='far1')

imp_rates = imp_matrix.sum(axis=1)

# %%

import matplotlib.pyplot as plt

plt.hist(gnd_matrix[:,1], 30)