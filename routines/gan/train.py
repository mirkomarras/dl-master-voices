#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

from helpers.dataset import get_mv_analysis_users, load_data_set, filter_by_gender
from helpers.datapipeline import data_pipeline_gan

from models import gan


def train_gan(model, dataset, gender, length=2.58, batch=32, examples=0, resize=None, epochs=500):
    output = 'spectrum'
    sample_rate = 16000
    slice_len = int(length * sample_rate)
    vox_meta = './data/vs_mv_pairs/meta_data_vox12_all.csv'
    vox_data = './data/vs_mv_pairs/data_mv_vox2_all.npz'
    gender = gender if gender is not 'neutral' else None

    # Load paths and labels for audio files that will be used during the optimization procedure
    audio_dir = map(str, dataset.split(','))
    mv_user_ids = get_mv_analysis_users(vox_data, type='train') # We retrieve the list of user IDs included in the training split of the mv optimization procedure
    x_train, y_train = load_data_set(audio_dir, mv_user_ids, include=False) # We load all the audio files and corresponding labels for the above-mentioned user IDs
    if gender:
        x_train, y_train = filter_by_gender(x_train, y_train, vox_meta, gender) # We keep only the audio files of users having the gender against which mv will be optimized

    if examples > 0:
        x_train = x_train[:examples]

    # Create and train model
    train_data = data_pipeline_gan(x_train, slice_len=slice_len, sample_rate=sample_rate, batch=batch, prefetch=1024, output_type=output, pad_width='auto', resize=resize)

    # Set output size for the GAN
    height = train_data.element_spec.shape[1]
    width = train_data.element_spec.shape[2]    
    width_ratio = width / height

    print('Training ' +  model + ' on ' + gender + ' (' + str(length) + 's clips)')

    print('> datasets: ' + dataset + ' - #Samples: ' + str(len(x_train)) + ' ' + str(train_data.element_spec.shape))
            
    if model == 'dc-gan':
        gan_ = gan.DCGAN(gender, patch=height, width_ratio=width_ratio)
    elif model == 'ms-gan':
        gan_ = gan.MultiscaleGAN(gender, patch=height, width_ratio=width_ratio, min_output=8)
    else:
        raise ValueError('Unsupported GAN model: ' + str(model))

    gan_.summarize_models()

    print('\n> saving results & models to ' + gan_.dirname())

    gan_.train(train_data, epochs, 10)
    gan_.save(True, False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GAN training')

    parser.add_argument('-m', '--model', dest='model', default='ms-gan', type=str, action='store', choices=['dc-gan', 'ms-gan'], help='Network model architecture')
    parser.add_argument('-g', '--gender', dest='gender', default='neutral', type=str, help='Gender, e.g.: neutral, male, female')
    parser.add_argument('-d', '--dataset', dest='dataset', default='./data/voxceleb1/dev,./data/voxceleb2/dev', type=str, help='Path to the voxceleb train directory')
    parser.add_argument('-l', '--length', dest='length', default=2.58, type=float, action='store', help='Speech length [s] - 2.58 is the default which yields spectrogram of size 256')
    parser.add_argument('-b', '--batch', dest='batch', default=32, type=int, action='store', help='Batch size')
    parser.add_argument('-s', '--size', dest='size', default=None, type=int, action='store', help='Output size (height) - used for resizing the samples')
    parser.add_argument('-n', '--n_examples', dest='examples', default=0, type=int, action='store', help='Number of training examples (defaults to 0 - use all)')
    parser.add_argument('-e', '--epochs', dest='epochs', default=500, type=int, action='store', help='Number of training epochs (defaults to 500)')
        
    args = parser.parse_args()
    
    train_gan(args.model, args.dataset, args.gender, args.length, args.batch, args.examples, args.size, args.epochs)
