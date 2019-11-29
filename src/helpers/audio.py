#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scipy.io.wavfile import read as wavread
import tensorflow as tf
import soundfile as sf
import numpy as np
import sys
import os

def decode_audio(fp, sample_rate=None):
    audio_sf, audio_sr = sf.read(fp)
    return audio_sf

def load_noise_paths(noise_dir):
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
    assert sample_rate > 0
    noise_cache = {}
    for noise_type, noise_files in noise_paths.items():
        for nf in noise_files:
            noise_cache[nf] = decode_audio(nf, sample_rate=sample_rate).reshape((-1, 1, 1))
    return noise_cache

def get_tf_spectrum(signal, sample_rate=16000, frame_size=0.025, frame_stride=0.01, num_fft=512):
    assert sample_rate > 0 and frame_size > 0 and frame_stride > 0 and frame_stride < frame_size and num_fft > 0

    signal.set_shape((None, None, 1))
    frames = tf.contrib.signal.frame(signal, frame_length=int(frame_size*sample_rate), frame_step=int(frame_stride*sample_rate), pad_end=True, name='frames', axis=1)
    hamming_window = tf.contrib.signal.hamming_window(int(frame_size*sample_rate), periodic=True)
    frames = frames * tf.reshape(hamming_window, [1, 1, -1, 1])
    frames = tf.pad(frames, [[0, 0], [0, 0], [0, int(num_fft) - int(frame_size*sample_rate)], [0, 0]])
    frames = tf.transpose(frames, perm=[0, 1, 3, 2])
    frames = tf.cast(frames, tf.complex64)

    magnitude_spectrum = tf.cast(tf.abs(tf.spectral.fft(frames, name='fft')), tf.float32)
    magnitude_spectrum = tf.transpose(magnitude_spectrum, perm=[0, 3, 1, 2])

    mean_tensor, variance_tensor = tf.nn.moments(magnitude_spectrum, axes=[1])
    std_dev_tensor = tf.math.sqrt(variance_tensor)
    normalized_spectrum = (magnitude_spectrum - tf.expand_dims(mean_tensor, 1)) / tf.maximum(tf.expand_dims(std_dev_tensor, 1), 1e-8)

    normalized_spectrum_shape = normalized_spectrum.get_shape().as_list()
    assert normalized_spectrum_shape[1] == num_fft and normalized_spectrum_shape[3] == 1

    return normalized_spectrum

def get_tf_mfccs(signal, sample_rate=16000, frame_size=0.025, frame_stride=0.01, num_fft=512, n_filters=24, lower_edge_hertz=80.0, upper_edge_hertz=8000.0):
    assert sample_rate > 0 and frame_size > 0 and frame_stride > 0 and frame_stride < frame_size and num_fft > 0

    signal = tf.ensure_shape(tf.squeeze(signal, axis=[2]), (None, None))
    stfts = tf.contrib.signal.stft(signal, frame_length=int(frame_size*sample_rate), frame_step=int(frame_stride*sample_rate), fft_length=num_fft)
    spectrograms = tf.abs(stfts)
    num_spectrogram_bins = stfts.shape[-1].value

    linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(n_filters, num_spectrogram_bins, sample_rate, lower_edge_hertz, upper_edge_hertz)
    mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)

    log_mel_spectrograms_shape = log_mel_spectrograms.get_shape().as_list()
    assert len(log_mel_spectrograms_shape) == 3 and log_mel_spectrograms_shape[2] == n_filters

    mean_tensor, variance_tensor = tf.nn.moments(log_mel_spectrograms, axes=[1])
    normalized_log_mel_spectrograms = log_mel_spectrograms - tf.expand_dims(mean_tensor, 1)

    normalized_log_mel_spectrograms_shape = normalized_log_mel_spectrograms.get_shape().as_list()
    assert len(normalized_log_mel_spectrograms_shape) == 3 and normalized_log_mel_spectrograms_shape[2] == n_filters

    return normalized_log_mel_spectrograms