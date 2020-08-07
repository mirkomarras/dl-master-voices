#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.metrics import roc_curve
import pandas as pd
import numpy as np
import argparse
import os

from helpers.datapipeline import data_pipeline_mv
from helpers.dataset import get_mv_analysis_users, load_data_set, filter_by_gender, load_mv_data

from models.verifier.thinresnet34 import ThinResNet34
from models.mv.model import SiameseModel
from models.verifier.resnet50 import ResNet50
from models.verifier.resnet34 import ResNet34
from models.verifier.xvector import XVector
from models.verifier.vggvox import VggVox
from models import gan

import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def tuneThreshold(scores, labels, target_fa=None):
    '''
    Given a list of similarity scores and verification labels (0:diff-user pair, 1:same-user pair), this function computes the verification threshold, FAR,
    and FRR at a given target_fa false acceptance level. If target_fa=None, this function computes threshold, FAR, and FAR at the equal error rate.
    :param scores:      List of similarity scores
    :param labels:      List of verification labels associated to the similarity scores
    :param target_fa:   Target false acceptance level (if it is None, the EER level is considered)
    :return:            Threshold, FAR, and FRR at the target_fa level
    '''
    far, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    frr = 1 - tpr
    frr = frr*100
    far = far*100
    if target_fa:
        idx = np.nanargmin(np.absolute((target_fa - far)))
        return thresholds[idx], far[idx], frr[idx]
    idxE = np.nanargmin(np.absolute((frr - far)))
    return thresholds[idxE], far[idxE], frr[idxE]

def main():
    parser = argparse.ArgumentParser(description='Master voice training')

    # Parameters for verifier
    parser.add_argument('--netv', dest='netv', default='', type=str, action='store', help='Speaker verification model, e.g., xvector/v000')
    parser.add_argument('--policy', dest='policy', default='avg', type=str, choices=['avg','any'], action='store', help='Speaker verification policy')

    # Parameters for generative adversarial model or of the seed voice
    # Note: if netg is specified, the master voices will be created by sampling spectrums from the GAN; otherwise, you need to specify a seed voice to optimize as a master voice
    parser.add_argument('--netg', dest='netg', default=None, type=str, action='store', help='Generative adversarial model, e.g., ms-gan/v000')
    parser.add_argument('--netg_gender', dest='netg_gender', default='neutral', type=str, choices=['neutral', 'male', 'female'], action='store', help='Training gender of the generative adversarial model')
    parser.add_argument('--seed_voice', dest='seed_voice', default='', type=str, action='store', help='Path to the seed voice that will be optimized to become a master voice')

    # Parameters for master voice optimization
    parser.add_argument('--audio_dir', dest='audio_dir', default='./data/voxceleb2/dev', type=str, action='store', help='Path to the folder where master voice training audios are stored')
    parser.add_argument('--audio_meta', dest='audio_meta', default='./data/ad_voxceleb12/vox12_meta_data.csv', type=str, action='store', help='Path to the CSV file with id-gender metadata of master voice training audios')
    parser.add_argument('--mv_splits', dest='mv_splits', default='./data/ad_voxceleb12/vox2_mv_data.npz', type=str, action='store', help='Numpy file with master voice analysis paths and labels')
    parser.add_argument('--mv_gender', dest='mv_gender', default='female', type=str, choices=['neutral', 'male', 'female'], action='store', help='Geneder against which master voices will be optimized')
    parser.add_argument('--n_examples', dest='n_examples', default=100, type=int, action='store', help='Number of master voices sampled to be created (only if netg is set)')
    parser.add_argument('--n_epochs', dest='n_epochs', default=1024, type=int, action='store', help='Number of optimization epochs for each master voice example')
    parser.add_argument('--batch', dest='batch', default=128, type=int, action='store', help='Size of a training batch against which master voices will be optimized')
    parser.add_argument('--prefetch', dest='prefetch', default=1024, type=int, action='store', help='Number of training batches pre-fetched by the data pipeline')
    parser.add_argument('--learning_rate', dest='learning_rate', default=1e-2, type=float, action='store', help='Learning rate for master voice perturbation')
    parser.add_argument('--n_templates', dest='n_templates', default=10, type=int, action='store', help='Number of enrolment samples per user used for impersonation testing')
    parser.add_argument('--sample_rate', dest='sample_rate', default=16000, type=int, action='store', help='Sample rate of the audio files')
    parser.add_argument('--n_seconds', dest='n_seconds', default=3, type=int, action='store', help='Length in seconds of an audio for master voice optimization')

    args = parser.parse_args()

    output_type = ('filterbank' if args.netv.split('/')[0] == 'xvector' else 'spectrum')

    # Parameter summary to print at the beginning of the script
    print('Parameters summary')

    print('>', 'Net Verifier: {}'.format(args.netv))
    print('>', 'Policy: {}'.format(args.policy))
    print('>', 'Output type: {}'.format(output_type))
    print('>', 'Sampling source: {}'.format('seed voice' if not args.netg else 'gan'))

    print('>', 'Net GAN: {}'.format(args.netg))
    print('>', 'Net GAN gender: {}'.format(args.netg_gender))
    print('>', 'Seed voice: {}'.format(args.seed_voice))

    print('>', 'Path to train-test audio: {}'.format(args.audio_dir))
    print('>', 'Path to train-test metadata: {}'.format(args.audio_meta))
    print('>', 'Path to train-test splits: {}'.format(args.mv_splits))
    print('>', 'Optimization gender: {}'.format(args.mv_gender))
    print('>', 'Number of master voice examples (only if netg is set): {}'.format(args.n_examples))
    print('>', 'Number of optimization epochs: {}'.format(args.n_epochs))
    print('>', 'Size of optimization batches: {}'.format(args.batch))
    print('>', 'Number of pre-fetched batches: {}'.format(args.prefetch))
    print('>', 'Optimization learning rate: {}'.format(args.learning_rate))
    print('>', 'Number of enrolment samples per user: {}'.format(args.n_templates))
    print('>', 'Sample rate of audio files: {}'.format(args.sample_rate))

    assert '/v' in args.netv

    # Load paths and labels for audio files that will be used during the optimization procedure
    audio_dir = map(str, args.audio_dir.split(','))
    mv_user_ids = get_mv_analysis_users(args.mv_splits, type='train') # We retrieve the list of user IDs included in the training split of the mv optimization procedure
    x_train, y_train = load_data_set(audio_dir, mv_user_ids, include=True) # We load all the audio files and corresponding labels for the above-mentioned user IDs
    x_train, y_train = filter_by_gender(x_train, y_train, args.audio_meta, args.mv_gender) # We keep only the audio files of users having the gender against which mv will be optimized

    print('Initializing siamese network for master voice optimization')
    # We build the paths to the subfolders within the ./data/vs_mv_data folder where seed and master voices will be saved
    # Ex. #1: if netv = 'vggvox/v003' and mv_gender = 'female', then  dir_sv = 'vggvox-v003_real_f_sv' and dir_mv = 'vggvox-v003_real_f_mv'
    # Ex. #2: if netv = 'vggvox/v003', netg = 'ms-gan/v001', netg_gender = 'female', and mv_gender = 'female', then  dir_sv = 'vggvox-v003_ms-gan-v001_f-f_sv' and dir_mv = 'vggvox-v003_ms-gan-v001_f-f_mv'
    dir_mv = os.path.join('.', 'data', 'vs_mv_data', args.netv.replace('/', '-') + ('_' + args.netg.replace('/', '-') + '_' + args.netg_gender[0] + '-' + args.mv_gender[0] if args.netg else '_real_u-' + args.mv_gender[0]) + '_mv')
    dir_sv = os.path.join('.', 'data', 'vs_mv_data', args.netv.replace('/', '-') + ('_' + args.netg.replace('/', '-') + '_' + args.netg_gender[0] + '-' + args.mv_gender[0] if args.netg else '_real_u-' + args.mv_gender[0]) + '_sv')
    # We initialize the siamese model that will be used to optimize master voices
    siamese_model = SiameseModel(sample_rate=args.sample_rate, dir_mv=dir_mv, dir_sv=dir_sv, params=args)
    print('> siamese network initialized')

    print('Setting verifier')
    # We initialize, build, and load a pre-trained speaker verification model; this model will be duplicated in order to create the siamese model
    available_nets = {'xvector': XVector, 'vggvox': VggVox, 'resnet50': ResNet50, 'resnet34': ResNet34, 'thin_resnet': ThinResNet34}
    selected_verifier = available_nets[args.netv.split('/')[0]](id=(int(args.netv.split('/')[1].replace('v','')) if '/v' in args.netv else -1))
    siamese_model.set_verifier(selected_verifier)
    print('> verifier set')

    if args.netg is not None:
        print('Setting generator')
        # If we want to create mv from GAN examples, we initialize, build, and load a pre-trained GAN into the siamese model
        available_generators = {'dc-gan': gan.DCGAN, 'ms-gan': gan.MultiscaleGAN}
        selected_generator = available_generators[args.netg.split('/')[0]](args.netg_gender, version=int(args.netg.split('/')[1].replace('v','')))
        siamese_model.set_generator(selected_generator)
        print('> generated set')

    print('Building siamese model')
    # We build a siamese model by duplicating the selected speaker verifier; hence, the siamese model will take two spectrograms as input and return the cosine similarity
    # between the speaker embeddings associated to those spectrograms. If netg is set, the right branch of the model will be fed with spectrograms generated by the GAN.
    siamese_model.build()
    print('> siamese model built')

    print('Checking data pipeline output')
    # We check the output of the master voice optimization pipeline, i.e., spectrograms extracted from the training audio files and their associated user labels
    train_data = data_pipeline_mv(x_train, y_train, args.sample_rate*args.n_seconds, args.sample_rate, args.batch, args.prefetch, output_type)

    for index, x in enumerate(train_data):
        print('>', index, x[0].shape, x[1].shape)
        if index == 10:
            break

    print('Retrieving verification thresholds to be used for impersonation rate computation')
    # In order to get EER and FAR1% verification thresholds, we load the similarity scores computed in Vox1-test verification trials pairs for the selected verifier
    vox1_test_results = pd.read_csv(os.path.join('./data/pt_models', args.netv, 'test_vox1_sv_test.csv'))
    vox1_test_results = vox1_test_results.loc[:, ~vox1_test_results.columns.str.contains('^Unnamed')]
    vox1_test_results.columns = ['label', 'score']
    thresholds = [tuneThreshold(vox1_test_results['score'].values, vox1_test_results['label'].values, target_fa)[0] for target_fa in [None, 1.0]]

    print('Optimizing master voice')
    train_data = data_pipeline_mv(x_train, y_train, args.sample_rate*args.n_seconds, args.sample_rate, args.batch, args.prefetch, output_type)
    test_data = load_mv_data(args.mv_splits, args.audio_dir.replace(args.audio_dir.split('/')[-1],''), args.audio_meta, args.sample_rate, args.n_templates)
    siamese_model.train(seed_voice=args.seed_voice, train_data=train_data, test_data=test_data, n_examples=args.n_examples, n_epochs=args.n_epochs, n_steps_per_epoch=len(x_train) // args.batch, thresholds=thresholds)

if __name__ == '__main__':
    main()