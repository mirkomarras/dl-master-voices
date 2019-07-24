#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 17:29:43 2019

@author: pkorus
"""
import sys
import struct
import numpy as np
from scipy.io import wavfile
from scipy.signal import medfilt
import webrtcvad

sys.path.append('..')

from utils import vad, filterbanks

def filterbank_vad(filterbanks, sample_rate, smoothing_window=63, threshold=1.05, med_kernel=31):
    
    # Check if the array is oriented as expected
    assert filterbanks.shape[0] > filterbanks.shape[1]

    # Number of filters in the bank
    nfilt = filterbanks.shape[1]
        
    # Find the bin which corresponds to the maximum voice frequency
    max_voice_freq = 2500
    voice_freq_mel = (2595 * np.log10(1 + (max_voice_freq) / 700))
    voice_low_freq_mel = (2595 * np.log10(1 + (200) / 700))
    
    # Define filter banks    
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    
    # Find the index of the bin closest to the voice frequency - it appears to be 18
    low_bin_index = np.argmin(np.abs(mel_points - voice_low_freq_mel))
    bin_index = np.argmin(np.abs(mel_points - voice_freq_mel))
    
#    print('VAD between bins {} - {}'.format(low_bin_index, bin_index))
    
    filterbanks -= np.quantile(filterbanks, 0.05) # filterbanks.min()
    filterbanks = filterbanks.clip(min=0)
        
    x = np.mean(filterbanks[:, low_bin_index:bin_index], axis=1)
    y = np.mean(filterbanks, axis=1)
    
    # Ratio of energy in low-pass band to total energy 
    z = x / y
    
    if smoothing_window > 0:
        w = np.hamming(smoothing_window)
        z = np.convolve(w/w.sum(), z, mode='same')
    
    return medfilt(z > threshold, med_kernel)
    

def google_vad(samples, sample_rate, window_duration_ms=30):
    
    if window_duration_ms not in [10, 20, 30]:
        raise ValueError('Unsupported window length (only 10, 20, 30 ms windows are allowed)')
        
    if sample_rate != 16000:
        raise ValueError('Code has been tested for 16kHz sampling')
    
    window_duration = window_duration_ms / 1000

    # Initialize the VAD model & set aggressive mode
    # Int [0,3] - 0 is the least aggressive about filtering out non-speech, 3 is the most aggressive. 
    vad_model = webrtcvad.Vad()    
    vad_model.set_mode(3)

    samples_per_window = int(window_duration * sample_rate + 0.5)
    bytes_per_sample = 2
    
    segments = []
    raw_samples = struct.pack("%dh" % len(samples), *samples)
    
    for start in np.arange(0, len(samples), samples_per_window):
        stop = min(start + samples_per_window, len(samples))
        
        # Sanity check for the final frame 
        if stop - start < samples_per_window:
            return segments
        
        try:
            is_speech = vad_model.is_speech(raw_samples[start * bytes_per_sample: stop * bytes_per_sample], sample_rate = sample_rate)
        except Exception as e:
            raise RuntimeError('Error processing frame starting at {}: {}'.format(start, e))
    
        segments.append(dict(
           start = start,
           stop = stop,
           is_speech = is_speech))
        
    return segments

def google_get_speech_samples(samples, sample_rate):
    segments = google_vad(samples, sample_rate)
    speech_samples = np.concatenate([ samples[segment['start']:segment['stop']] for segment in segments if segment['is_speech']])
    return speech_samples
    
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Load sample
    # ../../datasets/voxceleb/test/id10270/5r0dWxy17C8/00002.wav
    # ffmpeg -i 111.mp3 -acodec pcm_s16le -ac 1 -ar 16000 out.wav
    
    filename = '../tests/sample16kHz.wav'
    
    def compare_vad(filename):
    
        sample_rate, samples = wavfile.read(filename)
        
        # The Google VAD
        segments = google_vad(samples, sample_rate)
        
        # Simple VAD based on filterbank thresholding
        fb = filterbanks.filterbanks1d(samples, prefilter=True)
        vad_decisions = filterbank_vad(fb, sample_rate)
        
        # Plotting
        plt.figure(figsize = (18,7))
        plt.subplot(4,1,1)
        plt.plot(samples)
        plt.xlim([0, samples.shape[0]])
        
        ymax = max(samples)
        
        # plot segment identifed as speech
        for segment in segments:
            if segment['is_speech']:
                plt.plot([ segment['start'], segment['stop'] - 1], [ymax * 1.1, ymax * 1.1], color = 'orange')
        
        plt.xticks([])
        plt.yticks([])
        plt.ylabel('Google VAD')
        
        plt.subplot(4,1,2)
        plt.imshow(fb.T)
        plt.axis('auto')
    #    plt.yticks([])
        plt.xticks([])
        plt.xlim([0, fb.shape[0]])
        plt.plot([0, fb.shape[0]], [15, 15], 'w--')
        plt.ylabel('Mel filterbanks')
        
        plt.subplot(4,1,3)
        plt.plot(vad_decisions)
        plt.yticks([])
        plt.xticks([])
        plt.xlim([0, fb.shape[0]])
        plt.ylabel('Simple VAD')
        
        plt.subplot(4,1,4)
        plt.xlim([0, fb.shape[0]])
    
    #    filterbanks -= filterbanks.min()
        fb -= np.quantile(fb, 0.05) # filterbanks.min()
        fb = fb.clip(min=0)
                
        x = np.mean(fb[:, 2:15], axis=1)
        y = np.mean(fb, axis=1)
        
        w = np.hamming(15)
    #    w = np.ones((32,))
    #    x = np.convolve(w/w.sum(), x, mode='same')
    #    y = np.convolve(w/w.sum(), y, mode='same')
    
        z = x/y
        z = np.convolve(w/w.sum(), z, mode='same')
    
        plt.plot(x)
        plt.plot(y)
        plt.plot(z)
        plt.plot([0, fb.shape[0]], [1, 1], 'k:')
        plt.plot(medfilt(z > 1.05, 31))
    #    plt.yticks([])
        plt.xticks([])
        plt.ylabel('Ad-hoc play')
        
    compare_vad(filename)
    compare_vad('../../datasets/voxceleb/test/id10270/5r0dWxy17C8/00002.wav')
