#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import soundfile as sf
import numpy as np
import random
import math
import os

def decode_audio(fp, tgt_sample_rate=16000):
    """
    Function to decode an audio file
    :param fp:              File path to the audio sample
    :param tgt_sample_rate: Targeted sample rate
    :return:
    """

    assert tgt_sample_rate > 0

    audio_sf, audio_sr = sf.read(fp)

    if audio_sf.ndim > 1:
        print('>', 'warning: collapsing stereo into mono')
        audio_sf = audio_sf.mean(axis=-1)

    if audio_sr != tgt_sample_rate:
        print('>', 'warning: sampling frequency different than {:,d} Hz ({:,d}) -> audio naively downsampled'.format(tgt_sample_rate, audio_sr))
        audio_sf = audio_sf[::int(np.ceil(audio_sr / tgt_sample_rate))]

    return audio_sf

def load_noise_paths(noise_dir):
    """
    Function to load paths to noise audio samples
    :param noise_dir:       Directory path - organized in ./{speaker|room|microphone}/xyz.wav}
    :return:                Dictionary of paths to noise audio samples, e.g., noises['room'] = ['xyz.wav', ...]
    """

    assert os.path.exists(noise_dir)

    noise_paths = {}
    for noise_type in os.listdir(noise_dir):
        noise_paths[noise_type] = []
        for file in os.listdir(os.path.join(noise_dir, noise_type)):
            assert file.endswith('.wav')
            noise_paths[noise_type].append(os.path.join(noise_dir, noise_type, file))

        print('>', noise_type, len(noise_paths[noise_type]))

    return noise_paths

def cache_noise_data(noise_paths, sample_rate=16000):
    """
    Function to decode noise audio samples
    :param noise_paths:     Directory path - organized in ./{speaker|room|microphone}/xyz.wav} - returned by load_noise_paths(...)
    :param sample_rate:     Sample rate of an audio sample to be processed
    :return:                Dictionary of noise audio samples, e.g., cache['xyz.wav'] = [0.1, .54, ...]
    """

    assert sample_rate > 0

    noise_cache = {}
    for noise_type, noise_files in noise_paths.items():
        for nf in noise_files:
            noise_cache[nf] = decode_audio(nf, tgt_sample_rate=sample_rate).reshape((-1, 1, 1))

    return noise_cache

def get_tf_spectrum(signal, sample_rate=16000, frame_size=0.025, frame_stride=0.01, num_fft=1024):
    """
    Function to compute a tensorflow spectrum from signal
    :param signal:          Audio signal from which the spectrum will be extracted  - shape (None, 1)
    :param sample_rate:     Sample rate of an audio sample to be processed
    :param frame_size:      Size of a frame in seconds
    :param frame_stride:    Stride of a frame in seconds
    :param num_fft:         Number of FFT bins
    :return:                Spectrum - shape (None, num_fft / 2 + 1, None, 1)
    """

    assert sample_rate > 0 and frame_size > 0 and frame_stride > 0 and num_fft > 0
    assert frame_stride < frame_size

    signal = tf.squeeze(signal, axis=-1)

    # Compute the spectrogram
    magnitude_spectrum = tf.signal.stft(signal, int(frame_size*sample_rate), int(frame_stride*sample_rate), fft_length=num_fft)
    magnitude_spectrum = tf.abs(magnitude_spectrum)
    magnitude_spectrum = tf.transpose(magnitude_spectrum, perm=[0, 2, 1])
    magnitude_spectrum = tf.expand_dims(magnitude_spectrum, 3)

    # Normalize frames
    agg_axis = 2
    mean_tensor, variance_tensor = tf.nn.moments(magnitude_spectrum, axes=[agg_axis])
    std_dev_tensor = tf.math.sqrt(variance_tensor)
    normalized_spectrum = (magnitude_spectrum - tf.expand_dims(mean_tensor, agg_axis)) / tf.maximum(tf.expand_dims(std_dev_tensor, agg_axis), 1e-12)

    # Make sure the dimensions are as expected
    normalized_spectrum_shape = normalized_spectrum.get_shape().as_list()
    assert normalized_spectrum_shape[1] == num_fft / 2 + 1 and normalized_spectrum_shape[3] == 1

    return normalized_spectrum

def get_tf_filterbanks(signal, sample_rate=16000, frame_size=0.025, frame_stride=0.01, num_fft=512, n_filters=24, lower_edge_hertz=80.0, upper_edge_hertz=8000.0):
    """
    Function to compute tensorflow filterbanks from signal
    :param signal:              Audio signal from which the spectrum will be extracted  - shape (None, 1)
    :param sample_rate:         Sample rate of an audio sample to be processed
    :param frame_size:          Size of a frame in seconds
    :param frame_stride:        Stride of a frame in seconds
    :param num_fft:             Number of FFT bins
    :param n_filters:           Number of filters for the temporary log mel spectrum
    :param lower_edge_hertz:    Lower bound for frequencies
    :param upper_edge_hertz:    Upper bound for frequencies
    :return:                    Filterbanks - shape (None, None, n_filters)
    """

    assert sample_rate > 0 and frame_size > 0 and frame_stride > 0 and num_fft > 0
    assert frame_stride < frame_size

    # Compute the spectrogram
    signal = tf.squeeze(signal, axis=-1)
    magnitude_spectrum = tf.signal.stft(signal, int(frame_size*sample_rate), int(frame_stride*sample_rate), fft_length=num_fft)
    magnitude_spectrum = tf.abs(magnitude_spectrum)
    n_bins = magnitude_spectrum.shape[-1]

    # Compute the log mel spectrum
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(n_filters, n_bins, sample_rate, lower_edge_hertz, upper_edge_hertz)
    mel_spectrum = tf.tensordot(magnitude_spectrum, linear_to_mel_weight_matrix, 1)
    mel_spectrum.set_shape(magnitude_spectrum.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
    log_mel_spectrum = tf.math.log(mel_spectrum + 1e-6)

    # Normalize mfccs
    agg_axis = 2
    mean_tensor, variance_tensor = tf.nn.moments(log_mel_spectrum, axes=[agg_axis])
    std_dev_tensor = tf.math.sqrt(variance_tensor)
    normalized_log_mel_spectrum = (log_mel_spectrum - tf.expand_dims(mean_tensor, agg_axis)) / tf.maximum(tf.expand_dims(std_dev_tensor, agg_axis), 1e-12)

    # Make sure the dimensions are as expected
    normalized_log_mel_spectrum_shape = normalized_log_mel_spectrum.get_shape().as_list()
    assert normalized_log_mel_spectrum_shape[2] == n_filters

    return normalized_log_mel_spectrum

def play_n_rec(inputs, noises, cache, noise_strength='random'):
    """
    Function to add playback & recording simulation to a signal
    :param inputs:          Pair with the signals as first element and the impulse flags as second element
    :param noises:          Dictionary of paths to noise audio samples, e.g., noises['room'] = ['xyz.wav', ...]
    :param cache:           Dictionary of noise audio samples, e.g., cache['xyz.wav'] = [0.1, .54, ...]
    :param noise_strength:  Type of noise strenght to be applied to the speaker noise part - choices ['random']
    :return:                Audio signals with playback & recording simulation according to the impulse flags
    """

    signal, impulse = inputs

    output = signal

    if noises is not None and cache is not None:

        speaker = np.array(cache[random.choice(noises['speaker'])], dtype=np.float32)
        speaker_output = tf.nn.conv1d(tf.pad(output, [[0, 0], [0, tf.shape(speaker)[0]-1], [0, 0]], 'constant'), speaker, 1, padding='VALID')

        if noise_strength == 'random':
            noise_strength = tf.clip_by_value(tf.random.normal((1,), 0, 0.00001), 0, 10)

        noise_tensor = tf.random.normal(tf.shape(speaker_output), mean=0, stddev=noise_strength, dtype=tf.float32)
        speaker_output = tf.add(speaker_output, noise_tensor)

        speaker_flag = tf.math.multiply(speaker_output, tf.expand_dims(tf.expand_dims(impulse[:, 0], 1), 1))
        output_flag = tf.math.multiply(output, tf.expand_dims(tf.expand_dims(tf.math.abs(tf.math.subtract(impulse[:, 0],1)), 1), 1))
        output = tf.math.add(speaker_flag, output_flag)

        room = np.array(cache[random.choice(noises['room'])], dtype=np.float32)
        room_output = tf.nn.conv1d(tf.pad(output, [[0, 0], [0, tf.shape(room)[0]-1], [0, 0]], 'constant'), room, 1, padding='VALID')

        room_flag = tf.math.multiply(room_output, tf.expand_dims(tf.expand_dims(impulse[:, 1], 1), 1))
        output_flag = tf.math.multiply(output, tf.expand_dims(tf.expand_dims(tf.math.abs(tf.math.subtract(impulse[:, 1],1)), 1), 1))
        output = tf.math.add(room_flag, output_flag)

        microphone = np.array(cache[random.choice(noises['microphone'])], dtype=np.float32)
        microphone_output = tf.nn.conv1d(tf.pad(output, [[0, 0], [0, tf.shape(microphone)[0]-1], [0, 0]], 'constant'), microphone, 1, padding='VALID')

        microphone_flag = tf.math.multiply(microphone_output, tf.expand_dims(tf.expand_dims(impulse[:, 2], 1), 1))
        output_flag = tf.math.multiply(output, tf.expand_dims(tf.expand_dims(tf.math.abs(tf.math.subtract(impulse[:, 2],1)), 1), 1))
        output = tf.math.add(microphone_flag, output_flag)

    return output