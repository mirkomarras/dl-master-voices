#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 15:08:07 2021

@author: pkorus
"""
import tensorflow as tf

from models.verifier import xvector, vggvox, resnet50

# sv = xvector.XVector(id=0)
sv = vggvox.VggVox(id=0)
# sv = resnet50.ResNet50(id=0)
sv.build(0)
sv.calibrate_thresholds()
# sv.load(replace_model=False)
sv.infer()

# x.model.summary()
# x.build(5205)

# from models.verifier.model import VladPooling

# %%
# model_path = 'data/vs_mv_models/xvector/v000/model.h5'
model_path = 'data/vs_mv_models/vggvox/v000/model.h5'
# model_path = 'data/vs_mv_models/resnet50/v000/model.h5'
sv.model.load_weights(model_path, skip_mismatch=True, by_name=True)
sv.infer()

# x.model.load_weights(model_path)

# X = tf.keras.models.load_model(model_path, custom_objects={'VladPooling': VladPooling})

# %%

import numpy as np
import os
from matplotlib import pyplot as plt

from helpers.dataset import Dataset



# gallery = Dataset('data/vs_mv_pairs/mv_test_population_libri_100u_10s.csv')
# gallery = Dataset('data/vs_mv_pairs/mv_test_population_interspeech_1000u_1s.csv')
gallery = Dataset('data/vs_mv_pairs/mv_test_population_dev-test_100u_10s.csv')
gallery.precomputed_embeddings(sv)
    
mv_set = 'data/vs_mv_seed/female/'
    
filenames = [os.path.join(mv_set, file) for file in os.listdir(mv_set) if file.endswith('.wav')][:90]
# logger.info('retrieve master voice filenames {}'.format(len(filenames)))
    
embeddings = sv.predict(np.array(filenames))

# gallery.n_samples_per_person = 10

# %%

sim_matrix, imp_matrix, gnd_matrix = sv.test_error_rates(embeddings, gallery, policy='avg', level='far1')

imp_rates = imp_matrix.sum(axis=1) / 100

print(np.mean(imp_rates))

plt.subplot(2,2,1)
plt.hist(100 * imp_rates, np.linspace(0,100))
plt.xlim([0, 100])
plt.title(f'imp rate = {100 * np.mean(imp_rates):.2f} %')

plt.subplot(2,2,2)
plt.hist(sim_matrix.ravel(), 30)
for k in {'far1', 'eer'}:
    t = sv._thresholds[k]
    plt.plot([t, t], [0, plt.ylim()[-1]])
    

np.mean(sim_matrix.ravel() > sv._thresholds['far1'])
np.mean(sim_matrix > sv._thresholds['far1'], axis=-1).mean()

# %%

import matplotlib.pyplot as plt

plt.hist(imp_rates, 30)
plt.xlim([0, 100])

plt.hist(gnd_matrix[:,1], 30)


# %%

# sv.model.get_layer(name='embs').output
sv._inference_model.get_layer(name='embs').activation = None

embeddings = sv.predict(np.array(filenames))

plt.hist(embeddings.numpy().ravel(), 100)
plt.ylim([0, 1000])
plt.title(f'{sv.model.name} embeddings (100 seed female voices)')
