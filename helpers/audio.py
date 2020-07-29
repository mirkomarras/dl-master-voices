#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import soundfile as sf
import numpy as np
import librosa
import decimal
import random
import math
import os

def decode_audio(fp, sample_rate=16000):
    """
    Function to decode an audio file
    :param fp:              File path to the audio sample
    :param sample_rate: Targeted sample rate
    :return:                Audio sample
    """
    assert sample_rate > 0

    try:
        audio_sf, audio_sr = sf.read(fp, dtype='float32')
        if audio_sf.ndim > 1 or audio_sr != sample_rate:
            audio_sf, new_sample_rate = librosa.load(fp, sr=sample_rate, mono=True)
    except:
        audio_sf, new_sample_rate = librosa.load(fp, sr=sample_rate, mono=True)

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
    Function to cache noise audio samples
    :param noise_paths:     Directory path - organized in ./{speaker|room|microphone}/xyz.wav} - returned by load_noise_paths(...)
    :param sample_rate:     Sample rate of an audio sample to be processed
    :return:                Dictionary of noise audio samples, e.g., cache['xyz.wav'] = [0.1, .54, ...]
    """

    assert sample_rate > 0

    noise_cache = {}
    for noise_type, noise_files in noise_paths.items():
        for nf in noise_files:
            noise_cache[nf] = decode_audio(nf, sample_rate=sample_rate).reshape((-1, 1, 1))

    return noise_cache


def rolling_window(signal, window, step=1):
    '''
    Function to rolling a time window, i.e., creating frames from a signal
    :param signal:  Signal to be framed
    :param window:  Window size
    :param step:    Step size
    :return:
    '''
    shape = signal.shape[:-1] + (signal.shape[-1] - window + 1, window)
    strides = signal.strides + (signal.strides[-1],)
    return np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides)[::step]


def round_half_up(number):
    """
    Function to round half up a number
    :param number:      Number to be rounded
    :return:            Half-up-rounded number
    """
    return int(decimal.Decimal(number).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))


def deframesig(frames, siglen, frame_len, frame_step, winfunc=lambda x: np.ones((x,))):
    '''
    Function to deframe a signal
    :param frames:      Original frames
    :param siglen:      Lenght of the signal
    :param frame_len:   Lenght of a frame
    :param frame_step:  Step between two consecutive frames
    :param winfunc:     Function to be applied to each window
    :return:            (signal)
    '''
    frame_len = round_half_up(frame_len)
    frame_step = round_half_up(frame_step)
    numframes = np.shape(frames)[0]
    assert np.shape(frames)[1] == frame_len, '"frames" matrix is wrong size, 2nd dim is not equal to frame_len'

    indices = np.tile(np.arange(0, frame_len), (numframes, 1)) + np.tile(np.arange(0, numframes * frame_step, frame_step), (frame_len, 1)).T
    indices = np.array(indices, dtype=np.int32)
    padlen = (numframes - 1) * frame_step + frame_len

    if siglen <= 0: siglen = padlen

    rec_signal = np.zeros((padlen,))
    window_correction = np.zeros((padlen,))
    win = winfunc(frame_len)

    for i in range(0, numframes):
        window_correction[indices[i, :]] = window_correction[indices[i, :]] + win + 1e-15  # add a little bit so it is never zero
        rec_signal[indices[i, :]] = rec_signal[indices[i, :]] + frames[i, :]

    rec_signal = rec_signal / window_correction
    return rec_signal[0:siglen]

    return frames * win


def framesig(signal, frame_len, frame_step, winfunc=lambda x: np.ones((x,)), stride_trick=True):
    """
    Function to frame a signal
    :param signal:      Spectrum to normalize
    :param frame_len:   Lenght of a frame
    :param frame_step:  Step between two consecutive frames
    :param winfunc:     Function to be applied to each window
    :param stride_trick:Flag to select whether applying rolling window or not
    :return:            Frame matrix
    """
    slen = len(signal)
    frame_len = int(round_half_up(frame_len))
    frame_step = int(round_half_up(frame_step))
    if slen <= frame_len:
        numframes = 1
    else:
        numframes = 1 + int(math.ceil((1.0 * slen - frame_len) / frame_step)) # LV

    padlen = int((numframes - 1) * frame_step + frame_len)

    zeros = np.zeros((padlen - slen,))
    padsignal = np.concatenate((signal, zeros))
    if stride_trick:
        win = winfunc(frame_len)
        frames = rolling_window(padsignal, window=frame_len, step=frame_step)
    else:
        indices = np.tile(np.arange(0, frame_len), (numframes, 1)) + np.tile(np.arange(0, numframes * frame_step, frame_step), (frame_len, 1)).T
        indices = np.array(indices, dtype=np.int32)
        frames = padsignal[indices]
        win = np.tile(winfunc(frame_len), (numframes, 1))

    return frames * win


def normalize_frames(spectrum, epsilon=1e-12):
    """
    Function to compute a numpy spectrum from signal
    :param spectrum:        Spectrum to normalize
    :param epsilon:         Maximum standard deviation
    :return:                Normalized spectrum
    """
    frames = []
    means = []
    stds = []
    for v in spectrum:
        means.append(np.mean(v))
        stds.append(np.std(v))
        frames.append((v - np.mean(v)) / max(np.std(v), epsilon))
    return np.array(frames), np.array(means), np.array(stds)

def denormalize_frames(spectrum, means, stds, epsilon=1e-12):
    '''
    Function to denormalize a spectrum
    :param spectrum:    Spectrum to denormalize
    :param means:       Pre-computed means for this spectrum
    :param stds:        Pre-computed std deviations for this spectrum
    :param epsilon:     Maximum standard deviation
    :return:
    '''
    return np.array([z * max(stds[i],epsilon) + means[i] for i, z in enumerate(spectrum)])

def get_np_spectrum(signal, sample_rate=16000, frame_size=0.025, frame_stride=0.01, num_fft=512, normalized=True, full=False):
    assert signal.ndim == 1, 'Only 1-dim signals supported'

    frames = framesig(signal, frame_len=int(frame_size * sample_rate), frame_step=int(frame_stride * sample_rate), winfunc=np.hamming)
    fft = abs(np.fft.fft(frames, n=num_fft))

    fft = fft[:-1, :(num_fft // 2)]
    if not normalized:
        return fft

    if not full:
        fft = fft[:, :(num_fft // 2)]

    fft_norm, fft_mean, fft_std = normalize_frames(fft.T)

    return fft_norm, fft_mean, fft_std


def get_tf_spectrum(signal, sample_rate=16000, frame_size=0.025, frame_stride=0.01, num_fft=512, normalized=True):
    """
    Function to compute a tensorflow spectrum from signal
    :param signal:          Audio signal from which the spectrum will be extracted  - shape (None, 1)
    :param sample_rate:     Sample rate of an audio sample to be processed
    :param frame_size:      Size of a frame in seconds
    :param frame_stride:    Stride of a frame in seconds
    :param num_fft:         Number of FFT bins
    :return:                Spectrum - shape (None, num_fft / 2, None, 1)
    """

    assert sample_rate > 0 and frame_size > 0 and frame_stride > 0 and num_fft > 0
    assert frame_stride < frame_size

    signal = tf.squeeze(signal, axis=-1)

    # Compute the spectrogram
    magnitude_spectrum = tf.signal.stft(signal, int(frame_size*sample_rate), int(frame_stride*sample_rate), fft_length=num_fft)
    magnitude_spectrum = tf.abs(magnitude_spectrum)
    magnitude_spectrum = tf.transpose(magnitude_spectrum, perm=[0, 2, 1])
    magnitude_spectrum = tf.expand_dims(magnitude_spectrum, 3)

    # Drop the last element to get exactly `num_fft` bins
    magnitude_spectrum = magnitude_spectrum[:, :-1, :, :]
    assert magnitude_spectrum.shape[1] == num_fft / 2 and magnitude_spectrum.shape[3] == 1

    if not normalized:
        return magnitude_spectrum

    # Normalize frames
    agg_axis = 2
    mean_tensor, variance_tensor = tf.nn.moments(magnitude_spectrum, axes=[agg_axis])
    std_dev_tensor = tf.math.sqrt(variance_tensor)
    normalized_spectrum = (magnitude_spectrum - tf.expand_dims(mean_tensor, agg_axis)) / tf.maximum(tf.expand_dims(std_dev_tensor, agg_axis), 1e-12)

    # Make sure the dimensions are as expected
    assert normalized_spectrum.shape[1] == num_fft / 2 and normalized_spectrum.shape[3] == 1

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

def get_play_n_rec_audio(inputs, noises, cache, noise_strength='random'):
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

def inv_stft(signal, sample_rate=16000, frame_size=0.025, frame_stride=0.01, num_fft=512):
    return np.fft.fft(framesig(signal, frame_len=frame_size*sample_rate, frame_step=frame_stride*sample_rate, winfunc=np.hamming), n=num_fft)

def inv_istft(spectrum, slice_length, sample_rate=16000, frame_stride=0.01, num_fft=512):
    return deframesig(np.fft.ifft(spectrum, n=num_fft), slice_length, frame_len=num_fft, frame_step=frame_stride*sample_rate, winfunc=np.hamming)

def spectrum_to_signal(spectrum, slice_length, iter=300, sample_rate=16000, frame_size=0.025, frame_stride=0.01, num_fft=512):
    inv_signal = np.random.randn(slice_length)

    for i in range(iter):
        inv_spectrum_angle = np.angle(inv_stft(inv_signal, sample_rate, frame_size, frame_stride, num_fft))
        inv_spectrum = spectrum * np.exp(1.0j * inv_spectrum_angle)
        inv_signal = inv_istft(inv_spectrum, slice_length, sample_rate, frame_stride, num_fft)
        error = np.sqrt(np.sum((spectrum - abs(inv_spectrum))**2) / spectrum.size)
        print('\rIteration: {}/{} - Error: {} '.format(i+1, iter, error), end='')
    print()

    return inv_signal
