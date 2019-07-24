from helpers.audioutils import get_fft_spectrum_unpack, get_fft_filterbank_unpack
from multiprocessing import Pool
import numpy as np
import random

class FilterbankGenerator():

    def __init__(self, list_IDs, labels, max_chunk_size, batch_size, shuffle, sample_rate, nfilt, noises, num_fft, frame_size, frame_stride, preemphasis_alpha, vad, aug, prefilter, normalize, n_proc=8):
        self.list_IDs = list_IDs
        self.labels = labels
        self.max_chunk_size = max_chunk_size
        self.nfilt = nfilt
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.noises = noises
        self.sample_rate = sample_rate
        self.num_fft = num_fft
        self.frame_size = frame_size
        self.frame_stride = frame_stride
        self.preemphasis_alpha = preemphasis_alpha
        self.vad = vad
        self.aug = aug
        self.prefilter = prefilter
        self.normalize = normalize
        self.pool = Pool(processes=n_proc)
        self.on_epoch_end()

    def len(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def getitem(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        list_Labels = [self.labels[k] for k in indexes]
        return self.data_generation(list_IDs_temp, list_Labels)

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def data_generation(self, list_IDs_temp, list_Labels):
        X = []
        y = []
        params = [[path, self.sample_rate, self.nfilt, self.noises, self.num_fft, self.frame_size, self.frame_stride, self.preemphasis_alpha, self.vad, self.aug, self.prefilter, self.normalize] for path in list_IDs_temp]
        filterbanks = np.array(self.pool.map(get_fft_filterbank_unpack, params))
        for _, (sp, label) in enumerate(zip(filterbanks,list_Labels)):
            try:
                X.append(sp)
                y.append(label)
            except:
                continue
        X = np.array(X)
        y = np.array(y)
        return X, y

class SpectrumGenerator():

    def __init__(self, list_IDs, labels, max_chunk_size, batch_size, shuffle, sample_rate, nfilt, noises, num_fft, frame_size, frame_stride, preemphasis_alpha, vad, aug, prefilter, normalize, n_proc=8):
        self.list_IDs = list_IDs
        self.labels = labels
        self.max_chunk_size = max_chunk_size
        self.nfilt = nfilt
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.noises = noises
        self.sample_rate = sample_rate
        self.num_fft = num_fft
        self.frame_size = frame_size
        self.frame_stride = frame_stride
        self.preemphasis_alpha = preemphasis_alpha
        self.vad = vad
        self.aug = aug
        self.prefilter = prefilter
        self.normalize = normalize
        self.pool = Pool(processes=n_proc)
        self.on_epoch_end()

    def len(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def getitem(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        list_Labels = [self.labels[k] for k in indexes]
        return self.data_generation(list_IDs_temp, list_Labels)

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def data_generation(self, list_IDs_temp, list_Labels):
        X = []
        y = []
        params = [[path, self.sample_rate, self.nfilt, self.noises, self.num_fft, self.frame_size, self.frame_stride, self.preemphasis_alpha, self.vad, self.aug, self.prefilter, self.normalize] for path in list_IDs_temp]
        spectrums = np.array(self.pool.map(get_fft_spectrum_unpack, params))
        for _, (sp, label) in enumerate(zip(spectrums,list_Labels)):
            try:
                X.append(sp)
                y.append(label)
            except:
                continue
        X = np.array(X)
        y = np.array(y)
        return X, y