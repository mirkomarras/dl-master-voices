#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import argparse

from models import gan


def preview_gan(model, gender, version, samples=32, patch=256, aspect=1):
    
    print('Sampling from ' + model + ' on ' + gender)
                            
    if model == 'dc-gan':
        gan_ = gan.DCGAN(gender, patch=patch, width_ratio=aspect)
    elif model == 'ms-gan':
        gan_ = gan.MultiscaleGAN(gender, version=version, patch=patch, width_ratio=aspect, min_output=8)
    else:
        raise ValueError('Unsupported GAN model: ' + model)

    print('GAN model directory: ' + gan_.dirname())
    gan_.load()
    gan_.summarize_models()

    print()

    fig = gan_.preview()
    fig.tight_layout()
    plt.show(block=True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GAN training')

    parser.add_argument('-m', '--model', dest='model', default='ms-gan', type=str, action='store', choices=['dc-gan', 'ms-gan'], help='Network model architecture')
    parser.add_argument('-g', '--gender', dest='gender', default='neutral', type=str, help='Gender, e.g.: neutral, male, female')
    parser.add_argument('-v', '--version', dest='version', default=0, type=int, action='store', help='Version of the pretrained model')
    parser.add_argument('-p', '--patch', dest='patch', default=256, type=int, action='store', help='Output size')
    parser.add_argument('-a', '--aspect', dest='aspect', default=1, type=float, action='store', help='Aspect ratio')
    parser.add_argument('-s', '--samples', dest='samples', default=32, type=int, action='store', help='Samples')
        
    args = parser.parse_args()
    
    preview_gan(args.model, args.gender, args.version, args.samples, args.patch, args.aspect)
