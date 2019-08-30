#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
import os
import librosa
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from models.vggvoxdp.model import Model
from helpers.datapipeline import spectrogram, mfcc, playback_n_recording, speech_only, clip
from helpers.datasetutils import getData
from helpers.generatorutils import FilterbankGenerator, SpectrumGenerator
from helpers.audioutils import get_fft_spectrum_unpack, get_fft_filterbank_unpack, get_fft_spectrum, load_wav


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
# gen = SpectrumGenerator(data['paths'], data['labels'], 300, batch, True, 16000, 24, noises, 512, 0.025, 0.01, 0.97, False, aug, True, True)


class SG(object):

    def __init__(self, files):
        self._files = files
        self._index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._index >= len(self._files) - 1:
            raise StopIteration()
        # self._index = (self._index + 1) % len(self._files)
        self._index = self._index + 1
        return self._files[self._index]


# gen = SG(data['paths'])

# for x in gen: print(x)

# %%

def generator(files):

    index = 0
    for r in range(16):
        # The files
        f_speech = files[index]
        f_speaker = random.sample(noises['ir_speaker'], 1)[0]
        f_room = random.sample(noises['ir_room'], 1)[0]
        f_mic = random.sample(noises['ir_mic'], 1)[0]

        index = (index + 1) % len(files)

        speech = load_wav(f_speech, 16000)
        speaker, _ = librosa.load(f_speaker, sr=16000, mono=True)
        room, _ = librosa.load(f_room, sr=16000, mono=True)
        mic, _ = librosa.load(f_mic, sr=16000, mono=True)

        speech = speech.reshape((1, -1, 1))
        speaker = speaker.reshape((-1, 1, 1))
        room= room.reshape((-1, 1, 1))
        mic = mic.reshape((-1, 1, 1))

        yield {'speech': speech, 'speaker': speaker, 'room': room, 'mic': mic}

    raise StopIteration()

for x in generator(data['paths']):
    for k, v in x.items():
        print(k, v.shape)

# %% Measure processing time

with tf.device('/cpu:0'):

    dataset = tf.data.Dataset.from_generator(
            lambda : generator(data['paths']),
            {'speech': tf.float32, 'speaker': tf.float32, 'room': tf.float32, 'mic': tf.float32}
            )
    
    # dataset = dataset.map(speech_only)
    dataset = dataset.map(playback_n_recording, num_parallel_calls=4)
    dataset = dataset.map(spectrogram, num_parallel_calls=4)
    # dataset = dataset.map(clip)
    # dataset = dataset.map(mfcc)
    dataset = dataset.map(lambda x: tf.squeeze(x, axis=0))
    dataset = dataset.batch(4)
    dataset = dataset.prefetch(1)
    
    # Get data from
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    
    t1 = datetime.now()
    with tf.Session() as sess:
        for i in range(4):
            x = sess.run(next_element)
            if isinstance(x, dict):
                print({k: v.shape for k, v in x.items()})
            else:
                print('>>', x.shape, x.min(), x.max())
    
    t2 = datetime.now()
    
    print('Elapsed {}'.format(t2-t1))

# model = Model()

# model.build_model(next_element, len(np.unique(data['labels'])), './tmp')
# model.train_model(gen, 1000, len(data['paths']) // batch,
#                   1e-4, 0.1, True, './tmp')

# %%

# plt.imshow(get_fft_spectrum(data['paths'][4], 16000))