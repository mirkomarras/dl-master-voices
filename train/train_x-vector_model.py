from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from helpers.generatorutils import FilterbankGenerator
from helpers.datasetutils import getData
import models.xvector.model as XVector
from scipy.signal import lfilter
import tensorflow as tf
import numpy as np
import webrtcvad
import argparse
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

'''*********************************************************************************************************************
                                                MAIN LOOP
*********************************************************************************************************************'''

def main():
    parser = argparse.ArgumentParser(description='X-Vector Generation')

    parser.add_argument('--mode', dest='mode', default='train', type=str, action='store', help='Usage mode for x-vector model [train/extract].')

    # Training parameters
    parser.add_argument('--data_source_vox1', dest='data_source_vox1', default='', type=str, action='store', help='Base VoxCeleb1 path of the training datasets')
    parser.add_argument('--data_source_vox2', dest='data_source_vox2', default='', type=str, action='store', help='Base VoxCeleb2 path of the training datasets')
    parser.add_argument('--n_epochs', dest='n_epochs', default=1024, type=int, action='store', help='Number of training epochs')
    parser.add_argument('--batch_size', dest='batch_size', default=32, type=int, action='store', help='Size of training batches')
    parser.add_argument('--learning_rate', dest='learning_rate', default=1e-1, type=float, action='store', help='Batch size for training')
    parser.add_argument('--shuffle', dest='shuffle', default=True, type=bool, action='store', help='Shuffling training data')
    parser.add_argument('--dropout_proportion', dest='dropout_proportion', default=0.1, type=float, action='store', help='Batch size for training')

    # Acoustic parameters
    parser.add_argument('--sample_rate', dest='sample_rate', default=16000, type=int, action='store', help='Audio sample rate.')
    parser.add_argument('--preemphasis', dest='preemphasis', default=0.97, type=float, action='store', help='Pre-emphasis alpha')
    parser.add_argument('--frame_stride', dest='frame_stride', default=0.01, type=float, action='store', help='Frame stride')
    parser.add_argument('--frame_size', dest='frame_size', default=0.025, type=float, action='store', help='Frame size')
    parser.add_argument('--num_fft', dest='num_fft', default=512, type=int, action='store', help='Number of FFT values')
    parser.add_argument('--min_chunk_size', dest='min_chunk_size', default=10, type=int, action='store', help='Minimum x-axis size of a filterbank')
    parser.add_argument('--max_chunk_size', dest='max_chunk_size', default=300, type=int, action='store', help='Maximum x-axis size of a filterbank')
    parser.add_argument('--aug', dest='aug', default=0, type=int, action='store', help='Augmentation mode [0:no aug, 1:aug any, 2:aug seq, 3:aug_prob]')
    parser.add_argument('--vad', dest='vad', default=False, type=bool, action='store', help='Voice activity detection mode')
    parser.add_argument('--prefilter', dest='prefilter', default=True, type=bool, action='store', help='Prefilter mode')
    parser.add_argument('--normalize', dest='normalize', default=True, type=bool, action='store', help='Normalization mode')
    parser.add_argument('--nfilt', dest='nfilt', default=24, type=int, action='store', help='Number of filterbanks')

    # Other parameters
    parser.add_argument('--noises_dir', dest='noises_dir', default='', type=str, action='store', help='Input noise directory for augmentation')
    parser.add_argument('--model_dir', dest='model_dir', default='', type=str, action='store', help='Output directory for the trained model')
    parser.add_argument('--print_interval', dest='print_interval', default=1, type=int, action='store', help='Printing interval for training steps')

    args = parser.parse_args()

    noises = {}
    for type in os.listdir(args.noises_dir):
        noises[type] = []
        for file in os.listdir(os.path.join(args.noises_dir, type)):
            noises[type].append(os.path.join(args.noises_dir, type, file))

    data = getData(args.data_source_vox1, args.data_source_vox2)
    training_generator = FilterbankGenerator(data['paths'], data['labels'], args.max_chunk_size, args.batch_size, args.shuffle, args.sample_rate, args.nfilt, noises, args.num_fft, args.frame_size, args.frame_stride, args.preemphasis, args.vad, args.aug, args.prefilter, args.normalize)

    model = XVector.Model()
    model.build_model(len(np.unique(data['labels'])), args.nfilt, args.model_dir)
    model.train_model(training_generator, args.n_epochs, len(data['paths']) // args.batch_size, args.learning_rate, args.dropout_proportion, args.print_interval, args.model_dir)

if __name__ == "__main__":
    main()


# python ./train/train_x-vector_model.py --data_source_vox1 "/beegfs/mm10572/voxceleb1" --data_source_vox2 "/beegfs/mm10572/voxceleb2" --aug 3 --vad True --noises_dir "./data/noise" --model_dir "./models/xvector/pre-trained"


