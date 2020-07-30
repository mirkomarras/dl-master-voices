#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scipy.spatial.distance import cosine
import tensorflow as tf
import pandas as pd
import argparse
import os

from helpers.audio import load_noise_paths, cache_noise_data, decode_audio, get_tf_spectrum, get_tf_filterbanks

from models.verifier.thinresnet34 import ThinResNet34
from models.verifier.resnet50 import ResNet50
from models.verifier.resnet34 import ResNet34
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
    parser.add_argument('--mv_base_path', dest='mv_base_path', default='./data', type=str, action='store', help='Trials base path for master voice analysis waveforms')
    parser.add_argument('--test_list', dest='test_list', default='./data/vs_mv_pairs/mv', type=str, action='store', help='Master voice population to be tested')
    parser.add_argument('--save_path', dest='save_path', default='./data/pt_models/', type=str, help='Path for model and logs')

    # Parameters for raw audio
    parser.add_argument('--sample_rate', dest='sample_rate', default=16000, type=int, action='store', help='Sample rate audio')
    parser.add_argument('--n_seconds', dest='n_seconds', default=3, type=int, action='store', help='Segment lenght in seconds')
    parser.add_argument('--noise_dir', dest='noise_dir', default='./data/vs_noise_data', type=str, action='store', help='Noise directory')

    args = parser.parse_args()

    print('Parameters summary')

    output_type = ('filterbank' if args.net.split('/')[0] == 'xvector' else 'spectrum')

    print('>', 'Net: {}'.format(args.net))
    print('>', 'Mode: {}'.format(output_type))

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
    available_nets = {'xvector': XVector, 'vggvox': VggVox, 'resnet50': ResNet50, 'resnet34': ResNet34, 'thin_resnet': ThinResNet34}
    model = available_nets[args.net.split('/')[0]](id=int(args.net.split('/')[1].replace('v','')))
    model.build(classes=0, mode='test')
    model.load()

    # Create save path
    result_save_path = os.path.join(args.save_path, args.net, 'mvcmp_any')
    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)

    # Compute similarity scores
    print('Compute similarity scores')
    extractor = model.infer()
    embs = {}
    if os.path.isdir(args.test_list):
        for mvset in os.listdir(args.test_list):
            for version in os.listdir(os.path.join(args.test_list, mvset)):
                for tfile in os.listdir(os.path.join(args.test_list, mvset, version)):
                    if not os.path.exists(os.path.join(result_save_path, mvset, version, tfile)):
                        print('> opening', os.path.join(args.test_list, mvset, version, tfile))
                        df_1 = pd.read_csv(os.path.join(args.test_list, mvset, version, tfile), names=['label', 'path1', 'path2', 'gender'])

                        fp_tfile = os.path.join(args.test_list, mvset, version, tfile)
                        sc = []
                        lab = []
                        for index, row in df_1.iterrows():
                            if row['path1'] in embs:
                                emb_1 = embs[row['path1']]
                            else:
                                audio_1 = decode_audio(os.path.join(args.mv_base_path, row['path1']))
                                audio_1 = audio_1.reshape((1, -1, 1))
                                inp_1 = get_tf_spectrum(audio_1) if output_type == 'spectrum' else get_tf_filterbanks(audio_1)
                                emb_1 = tf.keras.layers.Lambda(lambda emb1: tf.keras.backend.l2_normalize(emb1, 1))(extractor.predict(inp_1))
                                embs[row['path1']] = emb_1

                            if row['path2'] in embs:
                                emb_2 = embs[row['path2']]
                            else:
                                audio_2 = decode_audio(os.path.join(args.mv_base_path, row['path2']))
                                audio_2 = audio_2.reshape((1, -1, 1))
                                inp_2 = get_tf_spectrum(audio_2) if output_type == 'spectrum' else get_tf_filterbanks(audio_2)
                                emb_2 = tf.keras.layers.Lambda(lambda emb2: tf.keras.backend.l2_normalize(emb2, 1))(extractor.predict(inp_2))
                                embs[row['path2']] = emb_2

                            computed_score = 1 - cosine(emb_1, emb_2)

                            lab.append(row['label'])
                            sc.append(computed_score)
                            if (index + 1) % 10 == 0:
                                print('\r> pair', index + 1, '/', len(df_1.index), '-', mvset, version, tfile, '- ver score example', computed_score, end='')

                        print()

                        df = pd.DataFrame(list(zip(sc, lab)), columns=['score', 'label'])
                        df['path1'] = df_1['path1']
                        df['path2'] = df_1['path2']
                        df['gender'] = df_1['gender']

                        if not os.path.exists(os.path.join(result_save_path, mvset, version)):
                            os.makedirs(os.path.join(result_save_path, mvset, version))
                        df.to_csv(os.path.join(result_save_path, mvset, version, tfile), index=False)

                        print('> saved', fp_tfile, 'scores in', os.path.join(result_save_path, mvset, version, tfile))

                    else:

                        print('> skipped', os.path.join(result_save_path, mvset, version, tfile))

if __name__ == '__main__':
    main()