#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.metrics import roc_curve
import tensorflow as tf
import soundfile as sf
import pandas as pd
import numpy as np
import argparse
import librosa
import pickle
import os

from helpers.dataset import get_mv_analysis_users, load_data_set, filter_by_gender
from helpers.audio import get_tf_spectrum, get_tf_filterbanks, decode_audio
from models.verifier.xvector import normalize_with_moments
from models.verifier.model import VladPooling

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'

def findThresholdAtFAR(far, value):
    return np.argmin(np.abs(value - far))

def getStats(identical, scores):
    far, tpr, thresholds = roc_curve(np.array(identical), np.array(scores))
    frr = 1 - tpr
    idx_eer = np.argmin(np.abs(far - frr))
    idx_far1 = findThresholdAtFAR(far, 0.01)
    return [round(thresholds[idx_eer], 4), round(far[idx_eer], 4), round(frr[idx_eer], 4), round(np.mean([frr[idx_eer], far[idx_eer]]), 4),
            round(thresholds[idx_far1], 4), round(far[idx_far1], 4), round(frr[idx_far1], 4), round(np.mean([frr[idx_far1], far[idx_far1]]), 4)]

def main():

    parser = argparse.ArgumentParser(description='Tensorflow speaker verification model training')

    # Parameters for verifier
    parser.add_argument('--nets', dest='nets', default='xvector/v001,vggvox/v001,resnet34vox/v001,resnet50vox/v001', type=str, action='store', help='Networks to be tested')
    parser.add_argument('--mv_base_path', dest='mv_base_path', default='./data/vs_voxceleb2/', type=str, action='store', help='Trials base path for master voice analysis waveforms')
    parser.add_argument('--mv_meta', dest='mv_meta', default='./data/ad_voxceleb12/vox2_mv_data.npz', type=str, action='store', help='Numpy data for master voice analysis metadata')
    parser.add_argument('--audio_meta', dest='audio_meta', default='./data/ad_voxceleb12/vox12_meta_data.csv', type=str, action='store', help='CSV file with id-gender metadata')
    parser.add_argument('--n_templates', dest='n_templates', type=int, default=10, action='store', help='Enrolment set size')
    parser.add_argument('--n_comparisons', dest='n_comparisons', type=int, default=100, action='store', help='Number of comparisons per model')

    args = parser.parse_args()

    print('Parameters summary')
    print('>', 'Networks', args.nets)
    print('>', 'Master voice base path: {}'.format(args.mv_base_path))
    print('>', 'Master voice meta path: {}'.format(args.mv_meta))
    print('>', 'Audio meta path: {}'.format(args.audio_meta))
    print('>', 'Number of samples per template: {}'.format(args.n_templates))
    print('>', 'Number of comparisons: {}'.format(args.n_comparisons))

    print('Loading models')
    models = {}
    model_names = args.nets.split(',')
    for model_name in model_names:
        print('> loading', model_name)
        base_model = tf.keras.models.load_model(os.path.join('./data/pt_models', model_name, 'model.h5'), custom_objects={'VladPooling': VladPooling, 'normalize_with_moments': normalize_with_moments, 'tf': tf})
        models[model_name] = tf.keras.Model(inputs=base_model.input, outputs=base_model.get_layer('fc7').output)

    print('Loading utterances')
    mv_user_ids = get_mv_analysis_users(args.mv_meta, type='train')
    x_train, y_train = load_data_set([args.mv_base_path], mv_user_ids, include=True)
    x_train_male, y_train_male = filter_by_gender(x_train, y_train, args.audio_meta, 'male')
    x_train_female, y_train_female = filter_by_gender(x_train, y_train, args.audio_meta, 'female')

    dict_gen = {'male': [], 'female': []}
    dict_all = {}

    for (x, y) in zip(x_train_female, y_train_female):
        if y not in dict_all:
            dict_all[y] = []
            dict_gen[y] = 'female'
        dict_all[y].append(x)

    for (x, y) in zip(x_train_male, y_train_male):
        if y not in dict_all:
            dict_all[y] = []
            dict_gen[y] = 'male'
        dict_all[y].append(x)

    print('>', len(list(dict_all.keys())), list(dict_gen.values()).count('female'), list(dict_gen.values()).count('male'))

    print('Individual test setup')
    setups = [(args.n_templates, 'any'), (args.n_templates, 'avg'), (1, 'raw')]

    for template_dim, policy in setups:
        print('>', policy, template_dim)
        for model_name, model in models.items():
            print('>>', model_name)
            mode = 'filterbanks' if 'xvector' in model_name else 'spectrum'
            identical = []
            scores = []
            genders = []
            for index in range(args.n_comparisons):
                user_1 = np.random.choice(list(dict_all.keys()))
                user_2 = np.random.choice(list(set(list(dict_all.keys())) - set([user_1])))

                path_1_1 = np.random.choice(dict_all[user_1])
                audio_1_1 = decode_audio(path_1_1).reshape((1, -1, 1))
                sp_1_1 = get_tf_spectrum(audio_1_1) if mode == 'spectrum' else get_tf_filterbanks(audio_1_1)
                emb_1_1 = model.predict(sp_1_1)

                paths_1_2 = np.random.choice(list(set(dict_all[user_1]) - set([path_1_1])), template_dim)
                embs_1_2 = []
                for path in paths_1_2:
                    audio = decode_audio(path).reshape((1, -1, 1))
                    sp = get_tf_spectrum(audio) if mode == 'spectrum' else get_tf_filterbanks(audio)
                    embs_1_2.append(model.predict(sp))

                paths_2 = np.random.choice(dict_all[user_2], template_dim)
                embs_2 = []
                for path in paths_2:
                    audio = decode_audio(path).reshape((1, -1, 1))
                    sp = get_tf_spectrum(audio) if mode == 'spectrum' else get_tf_filterbanks(audio)
                    embs_2.append(model.predict(sp))

                identical.append(1)
                genders.append(dict_gen[user_1])
                scores.append((tf.keras.layers.Dot(axes=1, normalize=True)([emb_1_1, np.mean(embs_1_2, axis=0)]))[0][0] if policy == 'avg' else (np.max([tf.keras.layers.Dot(axes=1, normalize=True)([emb_1_1, emb])[0] for emb in embs_1_2])))

                identical.append(0)
                genders.append(dict_gen[user_1])
                scores.append((tf.keras.layers.Dot(axes=1, normalize=True)([emb_1_1, np.mean(embs_2, axis=0)]))[0][0] if policy == 'avg' else (np.max([tf.keras.layers.Dot(axes=1, normalize=True)([emb_1_1, emb])[0] for emb in embs_2])))

                print('\r>>> ' + str(index + 1) + ' of ' + str(args.n_comparisons), '(', template_dim, policy, '):', getStats(identical, scores), end='')

            pd.DataFrame(list(zip(identical, scores, genders)), columns=['target', 'score', 'gender']).to_csv(os.path.join('./data/pt_models', model_name, 'test_vox2_mv_train_' + policy + '_' + str(template_dim) + '.csv'))
            print()

if __name__ == '__main__':
    main()