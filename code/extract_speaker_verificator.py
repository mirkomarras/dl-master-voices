from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from helpers.generatorutils import FilterbankGenerator, SpectrumGenerator
from helpers.datasetutils import getData
from helpers.coreutils import *
import models.xvector.model as XVector
from sklearn.metrics import roc_curve, auc
from helpers.audioutils import *
import models.resnet34vox.model as ResNet34Vector
import models.resnet50vox.model as ResNet50Vector
import models.vggvox.model as VGGVector
from scipy import spatial
from scipy.signal import lfilter
import tensorflow as tf
import pandas as pd
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
import csv

def extract_vecs(args, model, noises):
    paths = load_obj(args.test_paths)
    paths = [p.replace('../data', args.data_source) for p in paths]

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    with tf.Session(config=config) as sess:
        model.load_model(sess, args.model_dir)

        output_dict = np.empty((len(paths), args.embs_size))

        if os.path.exists(args.embs_path):
            print('Load pre-computed file')
            output_dict = np.load(args.embs_path)

        for p_index, path in enumerate(paths):

            if p_index >= args.start_index:
                print('Emb', p_index, '/', len(paths))
                f_1 = get_fft_filterbank(path, args.sample_rate, args.nfilt, noises, args.num_fft, args.frame_size, args.frame_stride, args.preemphasis, args.vad, args.aug, args.prefilter, args.normalize)
                output_dict[p_index] = model.get_emb(f_1, sess, args.min_chunk_size, args.max_chunk_size)
                np.save(args.embs_path, output_dict)


def main():
    parser = argparse.ArgumentParser(description='Speaker Verifier Evaluation')

    parser.add_argument('--verifier', dest='verifier', default='', type=str, action='store', help='Type of verifier [xvector|vggvox|resnet34vox|resnet50vox].')
    parser.add_argument('--noises_dir', dest='noises_dir', default='', type=str, action='store', help='Input noise directory for augmentation')

    # Evaluation parameters
    parser.add_argument('--data_source', dest='data_source', default='', type=str, action='store', help='Base path testing dataset')
    parser.add_argument('--model_dir', dest='model_dir', default='', type=str, action='store', help='Output directory for the trained model')
    parser.add_argument('--test_paths', dest='test_paths', default='', type=str, action='store', help='Input mv extraction')
    parser.add_argument('--embs_path', dest='embs_path', default='', type=str, action='store', help='Output embs path')
    parser.add_argument('--embs_size', dest='embs_size', default=512, type=int, action='store', help='Vector size')
    parser.add_argument('--start_index', dest='start_index', default=0, type=int, action='store', help='Starting index')

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

    args = parser.parse_args()

    noises = {}
    for type in os.listdir(args.noises_dir):
        noises[type] = []
        for file in os.listdir(os.path.join(args.noises_dir, type)):
            noises[type].append(os.path.join(args.noises_dir, type, file))

    model = None
    if args.verifier == 'xvector':
        model = XVector.Model()
    elif args.verifier == 'vggvox':
        model = VGGVector.Model()
    elif args.verifier == 'resnet34vox':
        model = ResNet34Vector.Model()
    elif args.verifier == 'resnet50vox':
        model = ResNet50Vector.Model()
    else:
        print('Unsupported verifier.')
        exit(1)

    extract_vecs(args, model, noises)

if __name__ == "__main__":
    main()

'''
$ python ./code/extract_speaker_verificator.py
  --verifier "xvector"
  --data_source "/beegfs/mm10572"
  --model_dir "./models/xvector/model"
  --embs_path "./data/vox2_embs/embs_xvector.csv"
  --embs_size 512
  --test_paths "./data/vox2_mv/test_vox2_abspaths_1000_users"
  --noises_dir "./data/noise"
  --start_index 0
  --sample_rate 16000
  --preemphasis 0.97
  --frame_stride 0.01
  --frame_size 0.025
  --num_fft 512
  --min_chunk_size 10
  --max_chunk_size 300
  --aug 0
  --vad False
  --prefilter True
  --normalize True
  --nfilt 24
'''


# python ./code/test_speaker_verificator.py --verifier "xvector" --data_source "/beegfs/mm10572/voxceleb1/test" --model_dir "./models/xvector/model" --noises_dir "./data/noise" --result_file "./data/vox1_eer/result_pairs_voxceleb1.csv" --trials_file "./data/vox1_eer/trial_pairs_voxceleb1.csv"


