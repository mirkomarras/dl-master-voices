#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 09:18:43 2021

@author: pkorus
"""

from tqdm import tqdm
import tensorflow as tf

from helpers.dataset import get_mv_analysis_users, load_data_set, filter_by_gender
from helpers.datapipeline import data_pipeline_generator_gan, data_pipeline_gan

from models import gan, ae

gan_ = ae.Autoencoder('voxceleb', z_dim=256, patch_size=256)

print('GAN model directory: ' + gan_.dirname())
gan_.load()
gan_.summarize_models()

# %%

dataset = 'voxceleb'
examples = 16

output = 'spectrum'
length=2.58
sample_rate = 16000
slice_len = int(length * sample_rate)


gender = dataset.split('-')[-1] if '-' in dataset else None
audio_dir = './data/voxceleb1/dev'
audio_dir = audio_dir.split(',')

x_train, y_train = load_data_set(audio_dir, {})

if gender is not None:
    x_train, y_train = filter_by_gender(x_train, y_train, vox_meta, gender)

if examples > 0:
    x_train = x_train[:examples]
    
# Create and train model
train_data = data_pipeline_gan(x_train, slice_len=slice_len, sample_rate=sample_rate, batch=16,
                               prefetch=1024, output_type=output, pad_width='auto', resize=False)

print(f'{dataset} dataset with {len(x_train)} samples [{train_data.element_spec.shape}]')

for x in train_data:
    print(x)

# %%

print()

fig = gan_.preview()
fig.tight_layout()
plt.show(block=True)