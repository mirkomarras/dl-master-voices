from sklearn.metrics import roc_curve, auc
from keras.models import load_model
from scipy.signal import lfilter
from scipy import spatial
import pandas as pd
import numpy as np
import decimal
import logging
import librosa
import random
import math
import csv
import os

def build_buckets(max_sec, step_sec, frame_step):
    buckets = {}
    frames_per_sec = int(1/frame_step)
    end_frame = int(max_sec*frames_per_sec)
    step_frame = int(step_sec*frames_per_sec)
    for i in range(0, end_frame+1, step_frame):
        s = i
        s = np.floor((s-7+2)/2) + 1  # conv1
        s = np.floor((s-3)/2) + 1  # mpool1
        s = np.floor((s-5+2)/2) + 1  # conv2
        s = np.floor((s-3)/2) + 1  # mpool2
        s = np.floor((s-3+2)/1) + 1  # conv3
        s = np.floor((s-3+2)/1) + 1  # conv4
        s = np.floor((s-3+2)/1) + 1  # conv5
        s = np.floor((s-3)/2) + 1  # mpool5
        s = np.floor((s-1)/1) + 1  # fc6
        if s > 0:
            buckets[i] = int(s)
    return buckets

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
                                               indices[i, :]] + win + 1e-15  # add a little bit so it is never zero
        rec_signal[indices[i, :]] = rec_signal[indices[i, :]] + frames[i, :]

    rec_signal = rec_signal / window_correction
    return rec_signal[0:siglen]

def magspec(frames, NFFT):
    if np.shape(frames)[1] > NFFT:
        logging.warn(
            'frame length (%d) is greater than FFT size (%d), frame will be truncated. Increase NFFT to avoid.',
            np.shape(frames)[1], NFFT)
    complex_spec = np.fft.rfft(frames, NFFT)
    return np.absolute(complex_spec)

def powspec(frames, NFFT):
    return 1.0 / NFFT * np.square(magspec(frames, NFFT))

def logpowspec(frames, NFFT, norm=1):
    ps = powspec(frames, NFFT)
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

def normalize_frames(m,epsilon=1e-12):
    return np.array([(v - np.mean(v)) / max(np.std(v),epsilon) for v in m])

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

def get_fft_spectrum(filename, params):
    signal = load_wav(filename,params['sample_rate'])
    frames = framesig(signal, frame_len=params['frame_len']*params['sample_rate'], frame_step=params['frame_step']*params['sample_rate'], winfunc=np.hamming)
    fft = abs(np.fft.fft(frames,n=params['num_fft']))
    fft_norm = normalize_frames(fft.T)
    return fft_norm

def findThresholdAtFAR(far, value):
    return np.argmin(np.abs(value - far))

all_embs = {}
def getDistance(file_1, file_2, model, params):
    if file_1 not in all_embs:
        original_file_1 = file_1
        file_1 = os.path.join(params['base_path'], file_1)
        sp_1 = get_fft_spectrum(file_1, params)
        emb_1 = np.squeeze(model.predict(sp_1.reshape(1, *sp_1.shape, 1)))
        all_embs[original_file_1] = emb_1
    else:
        emb_1 = all_embs[file_1]

    if file_2 not in all_embs:
        original_file_2 = file_2
        file_2 = os.path.join(params['base_path'], file_2)
        sp_2 = get_fft_spectrum(file_2, params)
        emb_2 = np.squeeze(model.predict(sp_2.reshape(1, *sp_2.shape, 1)))
        all_embs[original_file_2] = emb_2
    else:
        emb_2 = all_embs[file_2]

    return 1 - spatial.distance.cosine(emb_1, emb_2)

def getComparisonByFile(file, model, params):
    pairs = pd.read_csv(file, names=['type', 'frame_1', 'frame_2', 'gender'], sep=' ')

    print(pairs.head())

    identical = []
    similarities = []
    path1 = []
    path2 = []
    gender = []
    for index, row in pairs.iterrows():
        similarity = getDistance(row['frame_1'], row['frame_2'], model, params)
        path1.append(row['frame_1'])
        path2.append(row['frame_2'])
        gender.append(row['gender'])
        similarities.append(similarity)
        identical.append(row['type'])

    return identical, similarities, path1, path2, gender


if __name__ == '__main__':
    params = {'base_path': '/beegfs/mm11333/data',
              'model_name': '/beegfs/mm11333/dl-master-voices/data/pt_models/vggvox/v000/model.h5',
              'eval_name': '/beegfs/mm11333/dl-master-voices/data/vs_mv_pairs/mv',
              'max_sec': 10,
              'bucket_step': 1,
              'frame_step': 0.01,
              'sample_rate': 16000,
              'preemphasis_alpha': 0.97,
              'frame_len': 0.025,
              'num_fft': 512
    }

    model = load_model(params['model_name'])

    for file in os.listdir(params['eval_name'])[31:]:
        print(file)
        identical, distances, path1, path2, gender = getComparisonByFile(os.path.join(params['eval_name'], file), model, params)
        df = pd.DataFrame(list(zip(distances, identical, path1, path2, gender)), columns=['score', 'label', 'path1', 'path2', 'gender'])
        df.to_csv(os.path.join('/beegfs/mm11333/dl-master-voices/data/pt_models/vggvox/v000/mvcmp', file), index=False)