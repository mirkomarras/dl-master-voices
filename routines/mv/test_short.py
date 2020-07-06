#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import random
import os

from helpers.audio import load_noise_paths, cache_noise_data, decode_audio, get_tf_spectrum, get_tf_filterbanks
from helpers.dataset import load_test_data_from_file, load_mv_data, create_template_trials
from helpers.utils import save_obj

from models.verifier.resnet50vox import ResNet50Vox
from models.verifier.resnet34vox import ResNet34Vox
from models.verifier.xvector import XVector
from models.verifier.vggvox import VggVox

import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main():
    parser = argparse.ArgumentParser(description='Master voice testing')

    # Parameters for verifier
    parser.add_argument('--net', dest='net', default='', type=str, action='store', help='Network model architecture')

    # Parameters for master voice analysis
    parser.add_argument('--mv_base_path', dest='mv_base_path', default='/beegfs/mm11333/data', type=str, action='store', help='Trials base path for master voice analysis waveforms')
    parser.add_argument('--test_list', dest='test_list', default='/beegfs/mm11333/dl-master-voices/data/vs_mv_pairs/mv', type=str, action='store', help='Master voice population to be tested')
    parser.add_argument('--save_path', dest='save_path', default='./data/pt_models/', type=str, help='Path for model and logs')

    # Parameters for raw audio
    parser.add_argument('--sample_rate', dest='sample_rate', default=16000, type=int, action='store', help='Sample rate audio')
    parser.add_argument('--n_seconds', dest='n_seconds', default=3, type=int, action='store', help='Segment lenght in seconds')
    parser.add_argument('--noise_dir', dest='noise_dir', default='./data/vs_noise_data', type=str, action='store', help='Noise directory')

    args = parser.parse_args()

    print('Parameters summary')

    mode = ('filterbank' if args.net.split('/')[0] == 'xvector' else 'spectrum')

    print('>', 'Net: {}'.format(args.net))
    print('>', 'Mode: {}'.format(mode))

    print('>', 'Master voice base path: {}'.format(args.mv_base_path))
    print('>', 'Test list: {}'.format(args.test_list))
    print('>', 'Save path: {}'.format(args.save_path))

    print('>', 'Sample rate: {}'.format(args.sample_rate))
    print('>', 'Maximum number of seconds: {}'.format(args.n_seconds))
    print('>', 'Noise dir: {}'.format(args.noise_dir))

    # Load noise data
    print('Load impulse response paths')
    noise_paths = load_noise_paths(args.noise_dir)
    print('Cache impulse response data')
    noise_cache = cache_noise_data(noise_paths, sample_rate=args.sample_rate)

    # Create and restore model
    print('Creating model')
    available_nets = {'xvector': XVector, 'vggvox': VggVox, 'resnet50vox': ResNet50Vox, 'resnet34vox': ResNet34Vox}
    model = available_nets[args.net.split('/')[0]](id=int(args.net.split('/')[1].replace('v','')), noises=noise_paths, cache=noise_cache, n_seconds=args.n_seconds, sample_rate=args.sample_rate)
    model.load()

    # Create save path
    result_save_path = os.path.join(args.save_path, args.net, 'mvcmp_pb')
    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)

    # Compute similarity scores
    print('Compute similarity scores')
    embs = {}
    if os.path.isdir(args.test_list):
        for tfile in os.listdir(args.test_list):
            df_1 = pd.read_csv(os.path.join(args.test_list, tfile), names=['label', 'path1', 'path2', 'gender'], sep=' ')
            fp_tfile = os.path.join(args.test_list, tfile)
            sc = []
            lab = []
            for index, row in df_1.iterrows():
                if row['path1'] in embs:
                    emb_1 = embs[row['path1']]
                else:
                    audio_1 = decode_audio(os.path.join(args.mv_base_path, row['path1']))
                    audio_1 = audio_1.reshape((1, -1, 1))
                    inp_1 = get_tf_spectrum(audio_1) if mode == 'spectrum' else get_tf_filterbanks(audio_1)
                    emb_1 = tf.keras.layers.Lambda(lambda emb1: tf.keras.backend.l2_normalize(emb1, 1))(model.embed(inp_1))
                    embs[row['path1']] = emb_1

                noise_index = random.choice(range(len(noise_cache)))
                if row['path2'] + '_' + str(noise_index) in embs:
                    emb_2 = embs[row['path2'] + '_' + str(noise_index)]
                else:
                    audio_2 = decode_audio(os.path.join(args.mv_base_path, row['path2']))
                    noise_pb = np.squeeze(noise_cache[noise_index])
                    target_len = len(audio_2)
                    if len(noise_pb) < target_len:
                        noise_pb = np.pad(noise_pb, (0, target_len - len(noise_pb)), 'constant')
                    else:
                        noise_pb = noise_pb[:target_len]
                    audio_2 = np.add(audio_2, noise_pb)
                    audio_2 = audio_2.reshape((1, -1, 1))
                    inp_2 = get_tf_spectrum(audio_2) if mode == 'spectrum' else get_tf_filterbanks(audio_2)
                    emb_2 = tf.keras.layers.Lambda(lambda emb1: tf.keras.backend.l2_normalize(emb1, 1))(model.embed(inp_2))
                    embs[row['path2'] + '_' + str(noise_index)] = emb_2

                computed_score = float(tf.keras.layers.Dot(axes=1, normalize=True)([emb_1, emb_2])[0][0])

                lab.append(row['label'])
                sc.append(computed_score)
                if (index + 1) % 10 == 0:
                    print('\r> pair', index + 1, 'of', len(df_1.index), computed_score, end='')
            print()
            df = pd.DataFrame(list(zip(sc, lab)), columns=['score', 'label'])
            df['path1'] = df_1['path1']
            df['path2'] = df_1['path2']
            df['gender'] = df_1['gender']
            df.to_csv(os.path.join(result_save_path, tfile), index=False)
            print(fp_tfile)

if __name__ == '__main__':
    main()