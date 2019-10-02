#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
import os
import sys
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
t_batches = 3

if len(sys.argv) == 1:
    processing = 'verbatim'
else:
    processing = sys.argv[-1]

data = getData(sources)

noises = {}
for n_type in os.listdir(noise_dir):
    noises[n_type] = []
    for file in os.listdir(os.path.join(noise_dir, n_type)):
        noises[n_type].append(os.path.join(noise_dir, n_type, file))

print('Found impulse responses:')
for k, v in noises.items():
    print('  {} ({}) -> {:.60s}...'.format(k, len(v), str(v)))

# %%

cached = {}
for k, files in noises.items():
    for fi in files:
        cached[fi] = librosa.load(fi, sr=16000, mono=True)[0].reshape((-1, 1, 1))

def generator(files, labels):

    index = 0
    for r in range(len(files)):
        # The files
        f_speech = files[index]
        f_speaker = random.sample(noises['ir_speaker'], 1)[0]
        f_room = random.sample(noises['ir_room'], 1)[0]
        f_mic = random.sample(noises['ir_mic'], 1)[0]
        label = labels[index]

        index = (index + 1) % len(files)

        speech = load_wav(f_speech, 16000)
        speaker = cached[f_speaker]
        room = cached[f_room]
        mic = cached[f_mic]

        speech = speech.reshape((1, -1, 1))
        speech = speech[:, :48000, :]

        yield speech, label, speaker, room, mic

    raise StopIteration()

for x in generator(data['paths'][:5], data['labels'][:5]):
    print(x[0].shape, x[1], x[2].shape)

# %% MEASUREMENT ------------------------------------------------------------------------------------------------

config = tf.ConfigProto(allow_soft_placement=False, log_device_placement=True)
config.graph_options.rewrite_options.layout_optimizer = 2  # RewriterConfig.OFF    

run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()

if True:

    with tf.device('/cpu:0'):

        dataset = tf.data.Dataset.from_generator(
                lambda : generator(data['paths'], data['labels']),
                (tf.float32, tf.int32, tf.float32, tf.float32, tf.float32)
                )
        
        dataset = dataset.prefetch(batch)
        
        #dataset = dataset.map(clip)
        #dataset = dataset.batch(batch)

    #with tf.device('/cpu:0'):

        if processing == 'playback':
            dataset = dataset.map(playback_n_recording)
        elif processing == 'verbatim':
            dataset = dataset.map(speech_only)
        else:
            raise ValueError('ERROR Unknown processing!')

        dataset = dataset.map(spectrogram)
        dataset = dataset.map(lambda x, y: (tf.squeeze(x, axis=0), y))
        dataset = dataset.batch(batch)

        #dataset = dataset.prefetch(1)

    #with tf.device('/cpu:0'):
        
        # Get data from
        iterator = dataset.make_initializable_iterator()
        next_element = iterator.get_next()

    t1 = datetime.now()
    with tf.Session(config=config) as sess:

        sess.run(iterator.initializer)

        for i in range(t_batches):
            x = sess.run(next_element, options=run_options, run_metadata=run_metadata)
            
            if isinstance(x, dict):
                print({k: v.shape for k, v in x.items()})
            elif isinstance(x, tuple):
                print(*[v.shape if len(v.shape) > 1 else v for v in x])
            else:
                print('>>', x.shape, x.min(), x.max())

    t2 = datetime.now()

    print('Iteration time [{}/{} batches of {} elem.] {}'.format(processing, t_batches, batch, t2-t1))

# %% TRAINING ------------------------------------------------------------------------------------------------

tf.reset_default_graph()

#with tf.Graph().as_default(), tf.device('/cpu:0'):
 
with tf.device('/gpu:0'):

    dataset = tf.data.Dataset.from_generator(
            lambda : generator(data['paths'], data['labels']),
            (tf.float32, tf.int32, tf.float32, tf.float32, tf.float32)
            )
    
    dataset = dataset.prefetch(1024*16)
                
    if processing == 'playback':
        dataset = dataset.map(playback_n_recording, num_parallel_calls=4)
    elif processing == 'verbatim':
        dataset = dataset.map(speech_only)
    else:
        raise ValueError('ERROR Unknown processing!')    
    
    dataset = dataset.map(spectrogram, num_parallel_calls=4)
    dataset = dataset.map(lambda x, y: (tf.squeeze(x, axis=0), y))
    dataset = dataset.batch(batch)

with tf.device('/cpu:0'):
    
    # Get data from
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

model = Model(tf.get_default_graph())

model.build_model(*next_element, len(np.unique(data['labels'])), './tmp')
model.train_model(1000, len(data['paths']) // batch, 1e-4, 0.1, True, './tmp', iterator.initializer)

# %%

# plt.imshow(get_fft_spectrum(data['paths'][4], 16000))

