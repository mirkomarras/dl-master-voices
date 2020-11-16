#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse

import numpy as np
import tensorflow as tf

from helpers.dataset import get_mv_analysis_users, load_data_set, filter_by_gender
from helpers.datapipeline import data_pipeline_generator_gan, data_pipeline_gan

from models import gan


def train_gan(model, dataset, length=2.58, batch=32, examples=0, resize=None, epochs=500, dist='normal'):
    
    print(f'Training {model} on {dataset} ({length}s clips)')

    output = 'spectrum'
    sample_rate = 16000
    slice_len = int(length * sample_rate)
    vox_meta = './data/ad_voxceleb12/vox12_meta_data.csv'
    vox_data = './data/ad_voxceleb12/vox2_mv_data.npz'
    
    if dataset == 'mnist':
        (x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
        
        if examples > 0:
            x_train = x_train[:examples]
        x_train = x_train.reshape((-1, 28, 28, 1))
        x_train = x_train.astype(np.float32) / 255.0
        
        train_data = tf.data.Dataset.from_tensor_slices(x_train)
        train_data = train_data.padded_batch(batch, (32, 32, 1))

    if dataset == 'celeb-a':
        import imageio, skimage, tqdm
        data_dir = './data/celeb-a/'
        data_files = [os.path.join(data_dir, x) for x in os.listdir(data_dir)]

        if examples > 0:
            data_files = data_files[:examples]
        
        x_train = np.zeros((len(data_files), 128, 128, 1), dtype=np.float32)
        
        with tqdm.tqdm(total=len(data_files)) as pbar:
            for i, file in enumerate(data_files):
                image = imageio.imread(file)
                image = skimage.transform.resize(image, (128, 157))
                y1 = (157 - 128) // 2
                y2 = 157 - y1 - 1
                image = np.mean(image[:, y1:y2, :], axis=-1, keepdims=True) / 255.0
                x_train[i] = image
                pbar.update(1)
                                        
        train_data = tf.data.Dataset.from_tensor_slices(x_train)
        train_data = train_data.padded_batch(batch, (128, 128, 1))

    elif dataset.startswith('vctk'):
        import random
        
        if '-' in dataset:
            n_speakers = int(dataset.split('-')[-1])
        else:
            n_speakers = None
        
        audio_dir = './data/vctk/wave'
        
        dirs = os.listdir(audio_dir)
        if n_speakers is not None:
            dirs = dirs[:n_speakers]
            
        print(f'Using {len(dirs)} speakers from VCTK')
            
        x_train = []
        for d in dirs:
            dd = os.path.join(audio_dir, d)
            if os.path.isdir(dd):
                files = os.listdir(dd)
                x_train.extend(os.path.join(audio_dir, d, f) for f in files)
        
        random.shuffle(x_train)
        
        if examples > 0:
            x_train = x_train[:examples]
        
        train_data = data_pipeline_gan(x_train, slice_len=slice_len, sample_rate=sample_rate, batch=batch, 
                                       prefetch=1024, output_type=output, pad_width='auto', resize=resize)
        


    elif dataset == 'seven':
        audio_dir = './data/seven/train'
        x_train = [os.path.join(audio_dir, x) for x in os.listdir(audio_dir)]

        if examples > 0:
            x_train = x_train[:examples]
        
        train_data = data_pipeline_gan(x_train, slice_len=slice_len, sample_rate=sample_rate, batch=batch, 
                                       prefetch=1024, output_type=output, pad_width='auto', resize=resize)
        

    elif dataset == 'digits':
        audio_dir = './data/digits/train'
        x_train = [os.path.join(audio_dir, x) for x in os.listdir(audio_dir)]

        if examples > 0:
            x_train = x_train[:examples]
        
        train_data = data_pipeline_gan(x_train, slice_len=slice_len, sample_rate=sample_rate, batch=batch, 
                                       prefetch=1024, output_type=output, pad_width='auto', resize=resize)

    elif dataset.startswith('voxceleb'):
        gender = dataset.split('-')[-1] if '-' in dataset else None
        audio_dir = './data/voxceleb1/dev'
        audio_dir = audio_dir.split(',')
        
        x_train, y_train = load_data_set(audio_dir, {})

        if gender is not None:
            x_train, y_train = filter_by_gender(x_train, y_train, vox_meta, gender)
        
        if examples > 0:
            x_train = x_train[:examples]
            
        # Create and train model
        train_data = data_pipeline_gan(x_train, slice_len=slice_len, sample_rate=sample_rate, batch=batch,
                                       prefetch=1024, output_type=output, pad_width='auto', resize=resize)
    
    else:
        raise ValueError(f'Unsupported dataset: {dataset}')
        
    # Set output size for the GAN
    height = train_data.element_spec.shape[1]
    width = train_data.element_spec.shape[2]    
    width_ratio = width / height
        
    print(f'{dataset} dataset with {len(x_train)} samples [{train_data.element_spec.shape}]')        
            
    if model == 'dc-gan':
        gan_ = gan.DCGAN(dataset, patch=height, width_ratio=width_ratio, latent_dist=dist)
    elif model == 'ms-gan':
        gan_ = gan.MultiscaleGAN(dataset, patch=height, width_ratio=width_ratio, min_output=8, latent_dist=dist)
    else:
        raise ValueError(f'Unsupported GAN model: {model}')

    gan_.summarize_models()

    print(f'Saving results & models to {gan_.dirname()}')

    gan_.train(train_data, epochs, 10)
    gan_.save(True, False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GAN training')

    parser.add_argument('-m', '--model', dest='model', default='ms-gan', type=str, action='store', 
                        choices=['dc-gan', 'ms-gan'], help='Network model architecture')
    parser.add_argument('-d', '--dataset', dest='dataset', default='voxceleb', type=str, 
                        help='Dataset, e.g.: mnist, digits, voxceleb, voxceleb-male, voxceleb-female')
    parser.add_argument('-D', '--dist', dest='dist', default='normal', type=str, 
                        help='Latent space dist, e.g.: normal, uniform')
    parser.add_argument('-l', '--length', dest='length', default=2.58, type=float, action='store', 
                        help='Speech length [s] - 2.58 is the default which yields spectrogram of size 256')
    parser.add_argument('-b', '--batch', dest='batch', default=32, type=int, action='store', 
                        help='Batch size')
    parser.add_argument('-s', '--size', dest='size', default=None, type=int, action='store', 
                        help='Output size (height) - used for resizing the samples')
    parser.add_argument('-n', '--n_examples', dest='examples', default=0, type=int, action='store', 
                        help='Number of training examples (defaults to 0 - use all)')
    parser.add_argument('-e', '--epochs', dest='epochs', default=500, type=int, action='store', 
                        help='Number of training epochs (defaults to 500)')
        
    args = parser.parse_args()
    
    train_gan(args.model, args.dataset, args.length, args.batch, args.examples, args.size, args.epochs, args.dist)
