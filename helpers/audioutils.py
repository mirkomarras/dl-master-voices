from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy.signal import lfilter
import numpy as np
import webrtcvad
import logging
import decimal
import librosa
import random
import pickle
import struct
import queue
import math
import time
import os

def round_half_up(number):
    return int(decimal.Decimal(number).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))

def rolling_window(a, window, step=1):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)[::step]

def framesig(sig, frame_len, frame_step, winfunc=lambda x: np.ones((x,)), stride_trick=True):
    slen = len(sig)
    frame_len = int(round_half_up(frame_len))
    frame_step = int(round_half_up(frame_step))
    if slen <= frame_len:
        numframes = 1
    else:
        numframes = 1 + int(math.ceil((1.0 * slen - frame_len) / frame_step)) # LV

    padlen = int((numframes - 1) * frame_step + frame_len)

    zeros = np.zeros((padlen - slen,))
    padsignal = np.concatenate((sig, zeros))
    if stride_trick:
        win = winfunc(frame_len)
        frames = rolling_window(padsignal, window=frame_len, step=frame_step)
    else:
        indices = np.tile(np.arange(0, frame_len), (numframes, 1)) + np.tile(
            np.arange(0, numframes * frame_step, frame_step), (frame_len, 1)).T
        indices = np.array(indices, dtype=np.int32)
        frames = padsignal[indices]
        win = np.tile(winfunc(frame_len), (numframes, 1))

    return frames * win

def deframesig(frames, siglen, frame_len, frame_step, winfunc=lambda x: np.ones((x,))):
    frame_len = round_half_up(frame_len)
    frame_step = round_half_up(frame_step)
    numframes = np.shape(frames)[0]
    assert np.shape(frames)[1] == frame_len, '"frames" matrix is wrong size, 2nd dim is not equal to frame_len'

    indices = np.tile(np.arange(0, frame_len), (numframes, 1)) + np.tile(
        np.arange(0, numframes * frame_step, frame_step), (frame_len, 1)).T
    indices = np.array(indices, dtype=np.int32)
    padlen = (numframes - 1) * frame_step + frame_len

    if siglen <= 0: siglen = padlen

    rec_signal = np.zeros((padlen,))
    window_correction = np.zeros((padlen,))
    win = winfunc(frame_len)

    for i in range(0, numframes):
        window_correction[indices[i, :]] = window_correction[
                                               indices[i, :]] + win + 1e-15
        rec_signal[indices[i, :]] = rec_signal[indices[i, :]] + frames[i, :]

    rec_signal = rec_signal / window_correction
    return rec_signal[0:siglen]

def magspec(frames, NFFT):
    if np.shape(frames)[1] > NFFT:
        logging.warn('frame length (%d) is greater than FFT size (%d), frame will be truncated. Increase NFFT to avoid.',
            np.shape(frames)[1], NFFT)
    complex_spec = np.fft.rfft(frames, NFFT)
    return np.absolute(complex_spec)

def powspec(frames, NFFT):
    return 1.0 / NFFT * np.square(magspec(frames, NFFT))

def logpowspec(frames, NFFT, norm=1):
    ps = powspec(frames, NFFT);
    ps[ps <= 1e-30] = 1e-30
    lps = 10 * np.log10(ps)
    if norm:
        return lps - np.max(lps)
    else:
        return lps

def preemphasis(signal, coeff=0.95):
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])

def load_wav(filename, sample_rate):
    audio, sr = librosa.load(filename, sr=sample_rate, mono=True)
    audio = audio.flatten()
    return audio

def remove_dc_and_dither(sin, sample_rate):
    if sample_rate == 16e3:
        alpha = 0.99
    elif sample_rate == 8e3:
        alpha = 0.999
    else:
        print("Sample rate must be 16kHz or 8kHz only")
        exit(1)
    sin = lfilter([1,-1], [1,-alpha], sin)
    dither = np.random.random_sample(len(sin)) + np.random.random_sample(len(sin)) - 1
    spow = np.std(dither)
    sout = sin + 1e-6 * spow * dither
    return sout

def normalize_frames(m,epsilon=1e-12):
    frames = []
    means = []
    stds = []
    for v in m:
        means.append(np.mean(v))
        stds.append(np.std(v))
        frames.append((v - np.mean(v)) / max(np.std(v), epsilon))
    return np.array(frames), np.array(means), np.array(stds)

def denormalize_frames(m, means, stds, epsilon=1e-12):
    return np.array([z * max(stds[i],epsilon) + means[i] for i, z in enumerate(m)])

def min_max_frames(m):
    return 2 * (m - m.min())/(m.max() - m.min()) - 1

def augment_any(signal, sample_rate, noises):
    type = random.choice(list(noises.keys()))
    noise = random.choice(noises[type])
    noise_sig, noise_fs = librosa.load(noise, sr=sample_rate, mono=True)
    noise_sig = list(noise_sig)
    voc_sig = list(signal)
    while len(noise_sig) < len(voc_sig):
        noise_sig += noise_sig
    noise_sig = np.array(noise_sig)
    voc_sig = np.array(voc_sig)
    return voc_sig + noise_sig[:len(voc_sig)]

def augment_sequential(signal, sample_rate, noises):
    voc_sig = signal
    for key, value in noises.items():
        noise = random.choice(value)
        noise_sig, noise_fs = librosa.load(noise, sr=sample_rate, mono=True)
        voc_list = list(voc_sig)
        noise_sig = list(noise_sig)
        while len(noise_sig) < len(voc_list):
            noise_sig += noise_sig
        noise_sig = np.array(noise_sig)
        voc_sig = voc_sig + noise_sig[:len(voc_sig)]
    return voc_sig

def augment_prob(signal, sample_rate, noises):
    voc_sig = signal
    for key, value in noises.items():
        if random.choice([0,1]) == 1:
            noise = random.choice(value)
            noise_sig, noise_fs = librosa.load(noise, sr=sample_rate, mono=True)
            voc_list = list(voc_sig)
            noise_sig = list(noise_sig)
            while len(noise_sig) < len(voc_list):
                noise_sig += noise_sig
            noise_sig = np.array(noise_sig)
            voc_sig = voc_sig + noise_sig[:len(voc_sig)]
    return voc_sig

def vad_func(samples, sample_rate, window_duration_ms=30):
    if window_duration_ms not in [10, 20, 30]:
        raise ValueError('Unsupported window length (only 10, 20, 30 ms windows are allowed)')

    if sample_rate != 16000:
        raise ValueError('Code has been tested for 16kHz sampling')

    try:
        window_duration = window_duration_ms / 1000

        vad_model = webrtcvad.Vad()
        vad_model.set_mode(3)

        samples_per_window = int(window_duration * sample_rate + 0.5)
        bytes_per_sample = 2

        segments = []
        raw_samples = struct.pack("%dd" % len(samples), *samples)

        for start in np.arange(0, len(samples), samples_per_window):
            stop = min(start + samples_per_window, len(samples))
            if stop - start < samples_per_window:
                break
            try:
                is_speech = vad_model.is_speech(raw_samples[start * bytes_per_sample: stop * bytes_per_sample], sample_rate=sample_rate)
            except Exception as e:
                raise RuntimeError('Error processing frame starting at {}: {}'.format(start, e))
            segments.append(dict(start=start, stop=stop, is_speech=is_speech))

        speech_samples = np.concatenate([samples[segment['start']:segment['stop']] for segment in segments if segment['is_speech']])
        return speech_samples
    except:
        return samples

def get_fft_filterbank_unpack(args):
    return get_fft_filterbank(*args)

def get_fft_filterbank(filename, sample_rate, nfilt, noises, num_fft=512, frame_size=0.025, frame_stride=0.01, preemphasis_alpha=0.97, vad=False, aug=0, prefilter=True, normalize=True):
    signal = load_wav(filename, sample_rate)

    # Voice activity detection
    if vad:
        signal = vad_func(signal, sample_rate)

    assert aug <= 3, 'Only augmentation modes equal or less than 2 supported'

    max_signal_dim = sample_rate * 4
    if len(signal) > max_signal_dim:
        start = random.choice(range(len(signal) - max_signal_dim))
        signal = signal[start: start + max_signal_dim]

    # Augmentation
    if aug == 1:
        signal = augment_any(signal, sample_rate, noises)
    elif aug == 2:
        signal = augment_sequential(signal, sample_rate, noises)
    elif aug == 3:
        signal = augment_prob(signal, sample_rate, noises)

    assert signal.ndim == 1, 'Only 1-dim signals supported'

    # Pre-emphasis
    if prefilter:
        signal = np.append(signal[0], signal[1:] - preemphasis_alpha * signal[:-1])

    # Framing
    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
    signal_length = len(signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

    padded_signal_length = num_frames * frame_step + frame_length
    pad_signal = np.pad(signal, [(0, padded_signal_length - signal_length)], 'constant', constant_values=0)

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    # Hamming Window
    frames *= np.hamming(frame_length)

    # FFT
    NFFT = num_fft
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum

    # Filter banks
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])  # left
        f_m = int(bin[m])  # center
        f_m_plus = int(bin[m + 1])  # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)  # dB

    # Mean normalization
    if normalize:
        filter_banks -= (np.mean(filter_banks, axis=0) + 1e-8)

    filter_banks /= np.abs(filter_banks).max()

    if filter_banks.shape[0] > 300:
        filter_banks = filter_banks[:300, :]

    return filter_banks

def get_fft_spectrum_unpack(args):
    return get_fft_spectrum(*args)

def get_fft_spectrum(filename, sample_rate, nfilt, noises, num_fft=512, frame_size=0.025, frame_stride=0.01, preemphasis_alpha=0.97, vad=False, aug=0, prefilter=True, normalize=True):
    signal = load_wav(filename, sample_rate)

    # Voice activity detection
    if vad:
        signal = vad_func(signal, sample_rate)

    assert aug <= 3, 'Only augmentation modes equal or less than 2 supported'

    # Augmentation
    if aug == 1:
        signal = augment_any(signal, sample_rate, noises)
    elif aug == 2:
        signal = augment_sequential(signal, sample_rate, noises)
    elif aug == 3:
        signal = augment_prob(signal, sample_rate, noises)

    assert signal.ndim == 1, 'Only 1-dim signals supported'

    # get FFT spectrum
    frames = framesig(signal, frame_len=frame_size * sample_rate, frame_step=frame_stride * sample_rate, winfunc=np.hamming)
    fft = abs(np.fft.fft(frames,n=num_fft))
    fft_norm, fft_mean, fft_std = normalize_frames(fft.T)
    return fft_norm, fft_mean, fft_std
