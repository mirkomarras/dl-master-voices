#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 09:18:43 2021

@author: pkorus
"""

from tqdm import tqdm
import tensorflow as tf

import librosa

from helpers.dataset import get_mv_analysis_users, load_data_set, filter_by_gender
from helpers.datapipeline import data_pipeline_generator_gan, data_pipeline_gan

from helpers.audio import decode_audio, get_np_spectrum, denormalize_frames, spectrum_to_signal

from models import gan, ae

# %%

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# %%

gan_ = ae.VariationalAutoencoder('voxceleb', z_dim=256, patch_size=256)

print('GAN model directory: ' + gan_.dirname())
gan_.load()
gan_.summarize_models()

# %%

dataset = 'voxceleb'
examples = 16

output = 'spectrum'
length = 2.58
sample_rate = 16000
slice_len = int(length * sample_rate)


gender = dataset.split('-')[-1] if '-' in dataset else None
audio_dir = './data/voxceleb2/dev'
audio_dir = audio_dir.split(',')

x_train, y_train = load_data_set(audio_dir, {})

if gender is not None:
    x_train, y_train = filter_by_gender(x_train, y_train, vox_meta, gender)

if examples > 0:
    x_train = x_train[:examples]
    
# Create and train model
train_data = data_pipeline_gan(x_train, slice_len=slice_len, sample_rate=sample_rate, batch=1,
                               prefetch=1024, output_type=output, pad_width='auto', resize=None)

print(f'{dataset} dataset with {len(x_train)} samples [{train_data.element_spec.shape}]')

for x in train_data:
    print(x)

plotting.imsc(x.numpy(), cmap='jet')

# %%

print()

fig = gan_.preview()
fig.tight_layout()
plt.show(block=True)

# %% Invert sample

from helpers import plotting

plotting.imsc(x.numpy(), cmap='jet')
X = gan_.codec(x)

plotting.imsc(x, cmap='jet')
X = gan_.codec(x[np.newaxis, ..., np.newaxis])

plotting.imsc(X.numpy(), cmap='jet')

# inv_signal = spectrum_to_signal(X.numpy().T, slice_len)

sp = X.numpy().squeeze()

sp = np.vstack((sp, np.zeros((1, sp.shape[1])), sp[:0:-1]))
sp = sp.clip(0)

inv_signal = spectrum_to_signal(sp.T, int((sp.shape[1] + 1) / 100.0 * sample_rate), verbose=False)
librosa.output.write_wav('tmp/signal.wav', inv_signal, sample_rate)

plotting.imsc(sp, cmap='jet')

# %% 

aux_signal = decode_audio(x_train[0])[:slice_len-100]

# x, input_avg, input_std = get_np_spectrum(aux_signal.ravel(), normalized=False)
x = get_np_spectrum(aux_signal.ravel(), normalized=False)

sp = np.squeeze(np.squeeze(denormalize_frames(np.squeeze(x), input_avg, input_std)))

sp = np.vstack((sp, np.zeros((1, sp.shape[1])), sp[:0:-1]))
sp = sp.clip(0)

inv_signal = spectrum_to_signal(sp.T, int((sp.shape[1] + 1) / 100.0 * sample_rate), verbose=False)

librosa.output.write_wav('tmp/signal.wav', inv_signal, sample_rate)

# %% Distort latent space

m, lv = gan_.encode(x[np.newaxis, ..., np.newaxis])
z = gan_.reparameterize(m, lv)

z_ind = np.zeros(z.shape)
z_ind[0, 12] = 0.5

X = gan_.decode(z + z_ind)

plotting.imsc(X.numpy(), cmap='jet')