#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import os
import tensorflow as tf
from models.vggvox.model import Model
from helpers.datasetutils import getData
from helpers.generatorutils import FilterbankGenerator, SpectrumGenerator

aug = 0

noise_dir = './data/noise'
sources = ['./data/voxceleb1']
batch = 32

data = getData(sources)

noises = {}
for n_type in os.listdir(noise_dir):
    noises[n_type] = []
    for file in os.listdir(os.path.join(noise_dir, n_type)):
        noises[n_type].append(os.path.join(noise_dir, n_type, file))

# gen = FilterbankGenerator(data['paths'], data['labels'], 300, batch, True, 16000, 24, noises, 512, 0.025, 0.01, 0.97, False, aug, True, True)
gen = SpectrumGenerator(data['paths'], data['labels'], 300, batch, True, 16000, 24, noises, 512, 0.025, 0.01, 0.97, False, aug, True, True)

# tf.data.Dataset.from_generator

model = Model()

model.build_model(len(np.unique(data['labels'])), './tmp')
model.train_model(gen, 1000, len(data['paths']) // batch, 1e-4, 0.1, True, './tmp')
