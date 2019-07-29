from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from helpers.generatorutils import FilterbankGenerator, SpectrumGenerator
from helpers.datasetutils import getData
import models.xvector.model as XVector
import models.resnet34vox.model as ResNet34Vector
import models.resnet50vox.model as ResNet50Vector
import models.vggvox.model as VGGVector
from helpers.audioutils import *
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

def eval_any(args, model, noises, thresholds, enrol_size, trials, target_paths, target_ordered_labels, target_embs, target_males, target_females, utterance_per_person, file):
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    with tf.Session(config=config) as sess:
        results = []
        for n_thr, thr in enumerate(thresholds):
            mv_l_imp_male = []
            mv_l_imp_female = []

            for _ in range(trials):
                f_1 = get_fft_filterbank(file, args.sample_rate, args.nfilt, noises, args.num_fft, args.frame_size, args.frame_stride, args.preemphasis, args.vad, args.aug, args.prefilter, args.normalize)
                emb_mv = model.get_emb(f_1, sess, args.min_chunk_size, args.max_chunk_size)

                similarities_mv = np.zeros(len(target_paths))
                for i in range(len(target_paths)):
                    similarities_mv[i] = np.sum(np.square(emb_mv - target_embs[i]))

                mv_false_acceptance_count_per_person = np.zeros(len(target_ordered_labels))
                for u_index, u_label in enumerate(target_ordered_labels):
                    indexes = np.array(random.sample(range(utterance_per_person), enrol_size))
                    u_row = np.copy(similarities_mv)
                    u_row = u_row[(u_index * utterance_per_person):(u_index * utterance_per_person + utterance_per_person)]
                    u_row = u_row[indexes]
                    fac = len([1 for s in u_row if s < thr])
                    mv_false_acceptance_count_per_person[u_index] = fac

                imp_males = [index for index, s in enumerate(mv_false_acceptance_count_per_person) if s >= 1 and index * utterance_per_person in target_males]
                imp_females = [index for index, s in enumerate(mv_false_acceptance_count_per_person) if s >= 1 and index * utterance_per_person in target_females]
                mv_l_imp_male.append(len(imp_males))
                mv_l_imp_female.append(len(imp_females))

            item = [thr, file, round(np.mean(mv_l_imp_male) / (len(target_ordered_labels) / 2) * 100, 3), round(np.mean(mv_l_imp_female) / (len(target_ordered_labels) / 2) * 100, 3)]

            results.append(item)

        return results

def eval_avg(args, model, noises, thresholds, enrol_size, trials, target_paths, target_ordered_labels, target_embs, target_males,  target_females, utterance_per_person, file):
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    with tf.Session(config=config) as sess:
        results = []
        for n_thr, thr in enumerate(thresholds):

            mv_l_imp_male = []
            mv_l_imp_female = []

            for _ in range(trials):
                f_1 = get_fft_filterbank(file, args.sample_rate, args.nfilt, noises, args.num_fft, args.frame_size, args.frame_stride, args.preemphasis, args.vad, args.aug, args.prefilter, args.normalize)
                emb_mv = model.get_emb(f_1, sess, args.min_chunk_size, args.max_chunk_size)

                mv_false_acceptance_count_per_person = np.zeros(len(target_ordered_labels))
                for u_index, u_label in enumerate(target_ordered_labels):
                    indexes = np.array(random.sample(range(utterance_per_person), enrol_size))
                    u_embs = target_embs[(u_index * utterance_per_person):(u_index * utterance_per_person + utterance_per_person)]
                    u_embs = np.mean(u_embs[indexes], axis=0)
                    s = np.sum(np.square(emb_mv - u_embs))
                    mv_false_acceptance_count_per_person[u_index] = (1 if s < thr else 0)

                imp_males = [index for index, s in enumerate(mv_false_acceptance_count_per_person) if s >= 1 and index * utterance_per_person in target_males]
                imp_females = [index for index, s in enumerate(mv_false_acceptance_count_per_person) if s >= 1 and index * utterance_per_person in target_females]
                mv_l_imp_male.append(len(imp_males))
                mv_l_imp_female.append(len(imp_females))

            item = [thr,
                    file,
                    round(np.mean(mv_l_imp_male) / (len(target_ordered_labels) / 2) * 100, 3),
                    round(np.mean(mv_l_imp_female) / (len(target_ordered_labels) / 2) * 100, 3)
                    ]

            results.append(item)

        return results

def run_eval(args, model, noises):
    test_paths = load_obj(args.test_paths)
    test_labels = load_obj(args.test_labels)
    test_embs = np.load(args.test_embs)

    test_ordered_labels = np.unique(test_labels)

    user_id_position = 4
    test_male = []
    test_female = []
    vox_metadata = pd.read_csv(args.meta_file, header=None, names=['vid', 'vggid', 'gender', 'set'])
    for p_index, path in enumerate(test_paths):
        if (p_index+1) % 100 == 0:
            print('\rStep', p_index+1, '/', len(test_paths), end='')
        if vox_metadata.loc[vox_metadata.vid == path.split('/')[user_id_position], 'gender'].values[0] == 'm':
            test_male.append(p_index)
        else:
            test_female.append(p_index)

    mv_base_path = args.data_source
    any_eer_m = []
    any_eer_f = []
    any_far_m = []
    any_far_f = []
    avg_eer_m = []
    avg_eer_f = []
    avg_far_m = []
    avg_far_f = []

    n_voices = 0
    f_voices = []
    if os.path.isdir(mv_base_path):
        for index, file in enumerate(os.listdir(mv_base_path)):
            if args.mv_type in file and 'wav' in file:
                n_voices += 1
                f_voices.append(os.path.join(mv_base_path, file))
    else:
        n_voices = 1
        f_voices.append(mv_base_path)

    results = np.zeros((n_voices, 8))
    row = 0
    for index, file in enumerate(f_voices):

        r_any = eval_any(model=model, thresholds=[args.thr_eer, args.thr_far1], enrol_size=10, trials=1, target_paths=np.copy(test_paths), target_ordered_labels=np.copy(test_ordered_labels),
                 target_embs=np.copy(test_embs), target_males=np.copy(test_male), target_females=np.copy(test_female), utterance_per_person=100,
                 file=os.path.join(mv_base_path, file), args=args, noises=noises)
        any_eer_m.append(r_any[0][2])
        any_eer_f.append(r_any[0][3])
        any_far_m.append(r_any[1][2])
        any_far_f.append(r_any[1][3])

        r_avg = eval_avg(model=model, thresholds=[args.thr_eer, args.thr_far1], enrol_size=10, trials=1, target_paths=np.copy(test_paths), target_ordered_labels=np.copy(test_ordered_labels),
                 target_embs=np.copy(test_embs), target_males=np.copy(test_male), target_females=np.copy(test_female), utterance_per_person=100,
                 file=os.path.join(mv_base_path, file), args=args, noises=noises)
        avg_eer_m.append(r_avg[0][2])
        avg_eer_f.append(r_avg[0][3])
        avg_far_m.append(r_avg[1][2])
        avg_far_f.append(r_avg[1][3])

        print(' - - --- -------- | YEM  |  YEF  |  YFM  |  YFF  |  GEM  |  GEF  | GFM  |  GFF')
        print(row+1, '/', n_voices, file, ' | ',
              round(r_any[0][2],2), ' | ', round(r_any[0][3],2), ' | ', round(r_any[1][2],2), ' | ', round(r_any[1][3],2), ' | ',
              round(r_avg[0][2],2), ' | ', round(r_avg[0][3],2), ' | ', round(r_avg[1][2],2), ' | ', round(r_avg[1][3],2))
        print(row+1, '/', n_voices, file, ' | ',
              round(np.mean(any_eer_m),2), ' | ', round(np.mean(any_eer_f),2), ' | ', round(np.mean(any_far_m),2), ' | ', round(np.mean(any_far_f),2), ' | ',
              round(np.mean(avg_eer_m),2), ' | ', round(np.mean(avg_eer_f),2), ' | ', round(np.mean(avg_far_m),2), ' | ', round(np.mean(avg_far_f),2))

        temp = [r_any[0][2], r_any[0][3], r_any[1][2], r_any[1][3], r_avg[0][2], r_avg[0][3], r_avg[1][2], r_avg[1][3]]
        results[row] = np.array(temp)
        row += 1

    if os.path.isdir(mv_base_path):
        np.save(args.result_file + '.npy', results)

def main():
    parser = argparse.ArgumentParser(description='Master Voice Evaluation')

    parser.add_argument('--verifier', dest='verifier', default='', type=str, action='store', help='Type of verifier [xvector|vggvox|resnet34vox|resnet50vox].')
    parser.add_argument('--data_source', dest='data_source', default='', type=str, action='store', help='Dataset base path')
    parser.add_argument('--meta_file', dest='meta_file', default='', type=str, action='store', help='Dataset metadata')
    parser.add_argument('--result_file', dest='result_file', default='', type=str, action='store', help='Output result file')

    # Training parameters
    parser.add_argument('--test_paths', dest='test_paths', default='', type=str, action='store', help='MV test paths')
    parser.add_argument('--test_labels', dest='test_labels', default='', type=str, action='store', help='MV test labels')
    parser.add_argument('--test_embs', dest='test_embs', default='', type=str, action='store', help='MV test emeddings')
    parser.add_argument('--mv_set', dest='mv_set', default='', type=str, action='store', help='MV folder')

    # Impersonation parameters
    parser.add_argument('--mv_type', dest='mv_type', type=str, default='', action='store', help='MV name id')
    parser.add_argument('--thr_eer', dest='thr_eer', type=float, default=0, action='store', help='EER threshold')
    parser.add_argument('--thr_far1', dest='thr_far1', type=float, default=0, action='store', help='FAR1 threshold')
    parser.add_argument('--enrol_size', dest='enrol_size', type=int, default=10, action='store', help='Enrolment set size')
    parser.add_argument('--trials', dest='trials', type=int, default=1, action='store', help='Random verification trials')
    parser.add_argument('--utterance_per_person', type=int, dest='utterance_per_person', default=100, action='store', help='Test utterances per person')

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

    run_eval(args, model, noises)

if __name__ == "__main__":
    main()


'''
$ python ./code/eval_mv.py
  --verifier "xvector"
  --data_source "/beegfs/mm10572/voxceleb2"
  --noises_dir "./data/noise"
  --model_dir "./models/xvector/model"
  --result_file "./data/vox2_imp/result_imp_xvector.csv"
  --meta_file ""
  --test_paths ""
  --test_labels ""
  --test_embs ""
  --mv_set ""
  --mv_type "master"
  --thr_eer 200
  --thr_far1 2000
  --enrol_size 10
  --trials 1
  --utterance_per_person 100
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

# python ./train/train_speaker_verificator.py --verifier "xvector" --data_source_vox1 "/beegfs/mm10572/voxceleb1" --data_source_vox2 "/beegfs/mm10572/voxceleb2" --aug 3 --vad True --noises_dir "./data/noise" --model_dir "./models/xvector/model"


