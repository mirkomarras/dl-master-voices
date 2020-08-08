#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
 1. Input a GAN
 2. Generate N audio samples
 3. Create a folder in /data/vs_mv/data
 4. Save the audio samples under appropriate folder

'''

import soundfile as sf
import argparse

from helpers.audio import *
from models import gan

def generate_mv(model, gender, version, n=32, patch=256, aspect=1, sample_rate=16000):
    print('Sampling from ' + model + ' on ' + gender)

    if model == 'dc-gan':
        gan_ = gan.DCGAN(gender, patch=patch, width_ratio=aspect)
    elif model == 'ms-gan':
        gan_ = gan.MultiscaleGAN(gender, version=version, patch=patch, width_ratio=aspect, min_output=8)
    else:
        raise ValueError('Unsupported GAN model: ' + model)

    print('GAN model directory: ' + gan_.dirname())
    gan_.load()

    # Get the samples from the generator
    samples = gan_.sample(n).numpy()

    # Setup the output folder
    label = (gender[0]) + "-" + (gender[0])
    out_folder = "spectrogram_" + label + "_sv"
    gla_dir = os.path.join("./data/vs_mv_data/", out_folder, "v" + str("{:03d}".format(version)))
    if not os.path.exists(gla_dir):
        os.makedirs(gla_dir)

    # Use griffin-lim algorithm
    i = 0
    for s in samples:
        print("Sample #{}/{}".format(i+1,len(samples)))

        s2 = s.squeeze()

        # Mirror the spectrogram
        sp = np.vstack((s2, np.zeros((1, s2.shape[1])), s2[:0:-1]))

        # Clip the audio to remove noise
        sp = sp.clip(0)

        # Inverting
        inv_signal = spectrum_to_signal(sp.T, int(2.57*16000))

        # Export the audio
        sf.write(os.path.join(gla_dir, 'gen_' + gender + '_0' + (str(i)) + '.wav'), inv_signal, sample_rate)

        i += 1


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GAN training')

    parser.add_argument('-m', '--model', dest='model', default='ms-gan', type=str, action='store', choices=['dc-gan', 'ms-gan'], help='Network model architecture')
    parser.add_argument('-g', '--gender', dest='gender', default='neutral', type=str, help='Gender, e.g.: neutral, male, female')
    parser.add_argument('-v', '--version', dest='version', default=0, type=int, action='store', help='Version of the pretrained model')
    parser.add_argument('-p', '--patch', dest='patch', default=256, type=int, action='store', help='Output size')
    parser.add_argument('-a', '--aspect', dest='aspect', default=1, type=float, action='store', help='Aspect ratio')
    parser.add_argument('-s', '--samples', dest='samples', default=32, type=int, action='store', help='Samples')

    args = parser.parse_args()

    generate_mv(args.model, args.gender, args.version, args.samples, args.patch, args.aspect)
