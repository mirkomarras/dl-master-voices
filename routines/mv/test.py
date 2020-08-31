#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scipy.spatial.distance import cosine
import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import os

from helpers.audio import decode_audio, get_tf_spectrum, get_tf_filterbanks, get_play_n_rec_audio, load_noise_paths, cache_noise_data

from models.verifier.thinresnet34 import ThinResNet34
from models.verifier.resnet50 import ResNet50
from models.verifier.resnet34 import ResNet34
from models.verifier.xvector import XVector
from models.verifier.vggvox import VggVox

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main():
    parser = argparse.ArgumentParser(description='Tensorflow master voice trial pairs generation and testing')

    parser.add_argument('--net', dest='net', default='', type=str, action='store', help='Speaker model, e.g., vggvox/v003')
    parser.add_argument('--n_templates', dest='n_templates', default=10, type=int, action='store', help='Number of enrolled templates per user')
    parser.add_argument('--playback', dest='playback', default=0, type=int, action='store', help='Playback and recording in master voice test condition: 0 no, 1 yes')
    parser.add_argument('--mv_enrol', dest='mv_enrol', default='./data/vs_mv_pairs/trial_pairs_vox2_mv.csv', type=str, action='store', help='Path to the file with users enrolled templates')
    parser.add_argument('--noise_dir', dest='noise_dir', default='./data/vs_noise_data', type=str, action='store', help='Noise directory')

    args = parser.parse_args()

    assert args.net is not '', 'Please specify model network for --net'
    assert args.n_templates > 0, 'Please specify a number of templates per user greater than zero for --n_templates'
    assert args.mv_enrol is not '', 'Please specify a csv file with the enrolled templates for --mv_enrol'


    print('Parameters summary')
    print('>', 'Speaker model: {}'.format(args.net))
    print('>', 'Users enrolled templates: {}'.format(args.mv_enrol))
    print('>', 'Playback enabled? {}'.format('YES' if args.playback == 1 else 'NO'))
    print('>', 'Number of enrolled templates per user: {}'.format(args.n_templates))

    # Load noise data
    print('Load impulse response paths')
    noise_paths = load_noise_paths(args.noise_dir)
    print('Cache impulse response data')
    noise_cache = cache_noise_data(noise_paths)

    # Create the csv file with the trial verification pairs for each master voice, i.e., each master voice is compared with all the users enrolled templates
    mv_sets = [os.path.join(mv_set, version) for mv_set in os.listdir('./data/vs_mv_data') for version in os.listdir(os.path.join('./data/vs_mv_data', mv_set))]

    for mv_set in mv_sets: # Loop for each master voice set

        for mv_file in os.listdir(os.path.join('./data/vs_mv_data', mv_set)): # Loop for each master voice file

            if os.path.exists(os.path.join('data', 'vs_mv_pairs', 'mv', mv_set, mv_file.replace('.wav', '.csv'))): # We skip this master voice, if the csv file already exists
                continue

            if not mv_file.endswith('.wav'): # We skip all non-audio files
                continue

            print('> generating trial pairs for audio file', mv_file, 'stored in', os.path.join('./data/vs_mv_data', mv_set))
            df = pd.read_csv(args.mv_enrol, names=['label', 'path1', 'gender']) # Open the csv with the users enrolled templates
            df['path2'] = os.path.join('vs_mv_data', mv_set, mv_file) # As a second element in each trial pair, we put the current master voice

            if not os.path.exists(os.path.join('data', 'vs_mv_pairs', 'mv', mv_set)): # We will save the csv file within the vs_mv_pairs directory
                os.makedirs(os.path.join('data', 'vs_mv_pairs', 'mv', mv_set))

            df[['label', 'path1', 'path2', 'gender']].to_csv(os.path.join('data', 'vs_mv_pairs', 'mv', mv_set, mv_file.replace('.wav', '.csv')), index=False, header=False)
            print('> saved', mv_file, 'trial pairs in', os.path.join('data', 'vs_mv_pairs', 'mv', mv_set, mv_file.replace('.wav', '.csv')))

    # Create the csv file with the similarity scores for each master voice, i.e., each master voice is compared with all the users enrolled templates
    print('Compute similarity scores')
    nets = map(str, args.net.split(','))

    for net in nets:
        output_type = ('filterbank' if net.split('/')[0] == 'xvector' else 'spectrum')

        # Create main folder in ./data/vs_mv_models/ where the csv files with the similarity scores will be saved
        if not os.path.exists(os.path.join('./data/vs_mv_models/', net, 'mvcmp_any')):
            os.makedirs(os.path.join('./data/vs_mv_models/', net, 'mvcmp_any'))

        if not os.path.exists(os.path.join('./data/vs_mv_models/', net, 'mvcmp_avg')):
            os.makedirs(os.path.join('./data/vs_mv_models/', net, 'mvcmp_avg'))

        # Create and load speaker model
        print('Loading speaker model:', net)
        available_nets = {'xvector': XVector, 'vggvox': VggVox, 'resnet50': ResNet50, 'resnet34': ResNet34, 'thin_resnet': ThinResNet34}
        model = available_nets[net.split('/')[0]](id=int(net.split('/')[1].replace('v', '')))
        model.build(classes=0, mode='test')
        model.load()

        extractor = model.infer() # This returns a model that, given an input spectrum/filterbanks, returns the speaker embedding

        speaker_embs = {} # To speed up computation, we use a dictionary to save the speaker embeddings when they are loaded for the first time

        for mv_set in os.listdir('./data/vs_mv_pairs/mv'): # Loop for all the master voice sets

            if mv_set.startswith('.'):
                continue

            for version in os.listdir(os.path.join('./data/vs_mv_pairs/mv', mv_set)): # Loop for all the versions of each master voice set

                if version.startswith('.'):
                    continue

                for mv_csv_file in os.listdir(os.path.join('./data/vs_mv_pairs/mv', mv_set, version)): # Loop for all the csv files with the enrolled templates - master voice trial pairs

                    if mv_csv_file.startswith('.'):
                        continue

                    if os.path.exists(os.path.join('./data/vs_mv_models/', net, 'mvcmp_any', version, mv_csv_file)) and os.path.exists(os.path.join('./data/vs_mv_models/', net, 'mvcmp_avg', version, mv_csv_file)):
                        continue

                    print('> opening trial pairs', os.path.join('./data/vs_mv_pairs/mv', mv_set, version, mv_csv_file))
                    df_trial_pairs = pd.read_csv(os.path.join('./data/vs_mv_pairs/mv', mv_set, version, mv_csv_file), names=['label', 'path1', 'path2', 'gender'])

                    any_scores = [] # List of similarity scores for the any policy

                    avg_speaker_embs = [] # List of the embeddings for the current speaker
                    avg_speaker_files = [] # List of paths to the user templates of the current user
                    avg_speaker_sets = [] # List of paths to sets of user templates
                    avg_scores = [] # List of similarity scores for the avg policy
                    avg_gender = [] # List of genders for the speaker embeddings

                    for index, row in df_trial_pairs.iterrows():

                        if row['path1'] in speaker_embs: # If we already computed the embedding for the first element of the verification pair
                            emb_1 = speaker_embs[row['path1']]
                        else:
                            audio_1 = decode_audio(os.path.join('./data', row['path1'])).reshape((1, -1, 1)) # Load the user enrolled audio
                            input_1 = get_tf_spectrum(audio_1) if output_type == 'spectrum' else get_tf_filterbanks(audio_1) # Extract the acoustic representation
                            emb_1 = tf.keras.layers.Lambda(lambda emb1: tf.keras.backend.l2_normalize(emb1, 1))(extractor.predict(input_1)) # Get the speaker embedding
                            speaker_embs[row['path1']] = emb_1 # Save the current speaker embedding for future usage

                        if row['path2'] in speaker_embs: # If we already computed the embedding for the second element of the verification pair
                            emb_2 = speaker_embs[row['path2']]
                        else:
                            audio_2 = decode_audio(os.path.join('./data', row['path2'])).reshape((1, -1, 1)) # Load the master voice audio
                            if args.playback == 1:
                                print('> playback and recording simulated successfully')
                                audio_2 = get_play_n_rec_audio(audio_2, noise_paths, noise_cache, noise_strength='random') # Simulate playback and recording
                            input_2 = get_tf_spectrum(audio_2) if output_type == 'spectrum' else get_tf_filterbanks(audio_2) # Extract the acoustic representation
                            emb_2 = tf.keras.layers.Lambda(lambda emb2: tf.keras.backend.l2_normalize(emb2, 1))(extractor.predict(input_2)) # Get the speaker embedding
                            speaker_embs[row['path2']] = emb_2 # Save the current master voice speaker embedding for future usage

                        any_scores.append(1 - cosine(emb_1, emb_2)) # Compute the cosine similarity between the two embeddings
                        avg_speaker_embs.append(emb_1) # Add the current embedding to the list of embeddings of the current user
                        avg_speaker_files.append(row['path1']) # Add the current enrolled audio to the list of audio files of the current user

                        if (index + 1) % args.n_templates == 0: # When we analyze all the enrolled audio file for the current user
                            print('\r> pair', index + 1, '/', len(df_trial_pairs.index), '-', mv_set, version, mv_csv_file, end='')

                            # Compute cosine similarity between the averaged embedding and the master voice embedding
                            avg_scores.append(1 - cosine(np.average(avg_speaker_embs, axis=0), emb_2))
                            avg_speaker_sets.append((','.join(avg_speaker_files)))
                            avg_gender.append(row['gender'])

                            # Reset the avg 10 embedding list (even if not in use)
                            avg_speaker_embs, avg_speaker_files = [], []

                    print()

                    # Save the csv file for the any policy
                    mvcmp_any_dirname = os.path.join('./data/vs_mv_models/', net, 'mvcmp_any' + ('_playback' if args.playback == 1 else ''), mv_set, version)
                    mv_csv_file_with_scores = pd.DataFrame(list(zip(any_scores, df_trial_pairs['path1'], df_trial_pairs['path2'], df_trial_pairs['gender'])), columns=['score', 'path1', 'path2', 'gender'])
                    if not os.path.exists(mvcmp_any_dirname):
                        os.makedirs(mvcmp_any_dirname)
                    mv_csv_file_with_scores.to_csv(os.path.join(mvcmp_any_dirname, mv_csv_file), index=False)
                    print('> saved verification scores in', os.path.join(mvcmp_any_dirname, mv_csv_file))

                    # Save the csv file for the avg policy
                    mvcmp_avg_dirname = os.path.join('./data/vs_mv_models/', net, 'mvcmp_avg' + ('_playback' if args.playback == 1 else ''), mv_set, version)
                    mv_csv_file_with_scores = pd.DataFrame(list(zip(avg_scores, avg_speaker_sets, df_trial_pairs['path2'][:len(avg_gender)], avg_gender)), columns=['score', 'path1', 'path2', 'gender'])
                    if not os.path.exists(mvcmp_avg_dirname):
                        os.makedirs(mvcmp_avg_dirname)
                    mv_csv_file_with_scores.to_csv(os.path.join(mvcmp_avg_dirname, mv_csv_file), index=False)
                    print('> saved verification scores in', os.path.join(mvcmp_avg_dirname, mv_csv_file))


if __name__ == '__main__':
    main()

