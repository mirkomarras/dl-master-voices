# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp
import scipy.signal as sig
import scipy.io.wavfile
from scipy.fftpack import dct


def filterbanks1d(signal, *, sample_rate=16000, prefilter=True, normalize=True, nfilt=24, frame_size=0.025, frame_stride=0.01):
    
    assert signal.ndim == 1, 'Only 1-dim signals supported'
    
    if prefilter:
        pre_emphasis = 0.97
        emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
    else:
        emphasized_signal = signal
        
    # Framing        
    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame
    
    padded_signal_length = num_frames * frame_step + frame_length    
    pad_signal = np.pad(emphasized_signal, [(0, padded_signal_length - signal_length)], 'constant', constant_values=0)
    
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
        
    # Hamming Window
    frames *= np.hamming(frame_length)
        
    # FFT    
    NFFT = 512    
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum
        
    # Filter banks    
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)
    
    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right
    
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
    
    return filter_banks


def filterbanks(batch, sample_rate=16000, prefilter=True, normalize=True, nfilt=24, frame_size=0.025, frame_stride=0.01):
    """
    Computes mel-frequency filter-banks for a batch of waveforms. Expected tensor is Batch x Length
    """
        
    assert batch.ndim == 2, 'The batch should have 2 dimensions... (batch x signal length)'
    assert batch.shape[0] < batch.shape[1], 'Something is wrong - the batch_size seems greater than the signal length!'
    
    n_batch = batch.shape[0]
    
    # High-frequency emphasis
    if prefilter:
        pre_emphasis = 0.97
        for b in range(n_batch):
            batch[b] = np.convolve(np.pad(batch[b], [(1, 0)], 'edge'), [-pre_emphasis, 1], mode='valid')
                
    # Framing
    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
    signal_length = batch.shape[1]
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

    # Padding    
    padded_signal_length = num_frames * frame_step + frame_length    
    pad_signal = np.pad(batch, [(0, 0), (0, padded_signal_length - signal_length)], 'constant', constant_values=0)
    pad_signal = pad_signal.reshape((-1,1))

    #     
    indices = np.tile(np.arange(0, frame_length, dtype=np.int32), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step, dtype=np.int32), (frame_length, 1)).T
    indices = np.expand_dims(indices, axis=0) + np.arange(0, n_batch, dtype=np.int32).reshape((-1, 1, 1)) * padded_signal_length
    
    frames = pad_signal[indices].squeeze()
        
    # Window function    
    frames *= np.tile(np.hamming(frame_length).reshape((1,1,frame_length)), (n_batch, num_frames, 1))
            
    # FFT    
    NFFT = 512    
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum
            
    # Filter banks        
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)
    
    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    
    filter_banks = np.zeros((n_batch, num_frames, nfilt))
        
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right
    
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

    for b in range(n_batch):
        filter_banks[b] = np.dot(pow_frames[b], fbank.T)
        
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)  # dB
        
    # Mean normalization
    if normalize:
        filter_banks -= (np.mean(filter_banks, axis=1, keepdims=True) + 1e-8)
    
    # Normalize to -1 1
    filter_banks /= np.abs(filter_banks).max()
        
    assert filter_banks.shape == (n_batch, num_frames, nfilt), 'Something is wrong - invalid filter banks\' size'
    
    return filter_banks

