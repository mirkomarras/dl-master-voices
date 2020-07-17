import os
import argparse

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from helpers.dataset import get_mv_analysis_users, load_data_set, filter_by_gender
from helpers.datapipeline import data_pipeline_generator_gan, data_pipeline_gan

from models import gan


def preview_gan(model, dataset, version, samples=32, patch=256, aspect=1):
    
    print(f'Sampling from {model} on {dataset}')
                            
    if model == 'dc-gan':
        gan_ = gan.DCGAN(dataset, patch=patch, width_ratio=aspect)
    elif model == 'ms-gan':
        gan_ = gan.MultiscaleGAN(dataset, version=version, patch=patch, width_ratio=aspect, min_output=8)
    else:
        raise ValueError(f'Unsupported GAN model: {model}')

    print(f'GAN model directory: {gan_.dirname()}')
    gan_.load()
    gan_.summarize_models()
    fig = gan_.preview()
    fig.tight_layout()
    plt.show(block=True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GAN training')

    parser.add_argument('-m', '--model', dest='model', default='ms-gan', type=str, action='store', 
                        choices=['dc-gan', 'ms-gan'], help='Network model architecture')
    parser.add_argument('-d', '--dataset', dest='dataset', default='voxceleb', type=str, 
                        help='Dataset, e.g.: mnist, digits, voxceleb, voxceleb-male, voxceleb-female')
    parser.add_argument('-v', '--version', dest='version', default=0, type=int, action='store', 
                        help='Version of the pretrained model')
    parser.add_argument('-p', '--patch', dest='patch', default=256, type=int, action='store', 
                        help='Output size')
    parser.add_argument('-a', '--aspect', dest='aspect', default=1, type=float, action='store', 
                        help='Aspect ratio')
    parser.add_argument('-s', '--samples', dest='samples', default=32, type=int, action='store', 
                        help='Samples')
        
    args = parser.parse_args()
    
    preview_gan(args.model, args.dataset, args.version, args.samples, args.patch, args.aspect)
