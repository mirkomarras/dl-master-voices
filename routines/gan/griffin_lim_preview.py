import os
import argparse

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

# from helpers.audio import get_np_spectrum, denormalize_frames, spectrum_to_signal
# from helpers.dataset import get_mv_analysis_users, load_data_set, filter_by_gender
# from helpers.datapipeline import data_pipeline_generator_gan, data_pipeline_gan
from helpers.audio import *
from helpers import plotting

from models import gan


def griffin_lim_gan(model, dataset, version, n=32, patch=256, aspect=1, sample_rate=16000):
    
    print(f'Sampling from {model} on {dataset}')
                            
    if model == 'dc-gan':
        gan_ = gan.DCGAN(dataset, patch=patch, width_ratio=aspect)
    elif model == 'ms-gan':
        gan_ = gan.MultiscaleGAN(dataset, version=version, patch=patch, width_ratio=aspect, min_output=8)
    else:
        raise ValueError(f'Unsupported GAN model: {model}')

    print(f'GAN model directory: {gan_.dirname()}')
    gan_.load()

    #get the samples from the generator
    samples = gan_.sample(n).numpy()

    #print(samples)
    #print(samples[0].shape)
    #print(samples[0][:,0].shape)
    #print(samples[0].flatten().shape)

    #use griffin-lim algorithm
    i = 0
    for s in samples:
        print("SAMPLE #{}/{}".format(i+1,len(samples)))

        #CONVERTING (gives weird spectrogram)
        #norm, avg, std = tf_samples = get_np_spectrum(s[s.sum(axis=1).argmax(),:].flatten(), sample_rate=sample_rate,num_fft=512,full=True)
        #sp = denormalize_frames(norm, avg, std)
        s2 = s.squeeze()

        gla_dir = os.path.join(gan_.dirname(make=False), "gla_samples")
        if not os.path.exists(gla_dir):
            os.makedirs(gla_dir)

        sp = np.vstack((s2, np.zeros((1, s2.shape[1])), s2[:0:-1]))

        #print(sp)
        sp = sp.clip(0)
        #print(sp)

        #sp = sp - sp.min()

        f2 = plotting.imsc(sp)
        f2.savefig(os.path.join(gla_dir, f'test_audio' + (str(i)) + '.png'))
        #print(sp)

        
        
        #INVERTING
        inv_signal = spectrum_to_signal(sp.T, int(2.57*16000))

        print(inv_signal)

        #EXPORTING
        #fig = plotting.imsc(s, cmap='hsv')
        fig = plotting.imsc(sp, cmap='hsv')
        
        sf.write(os.path.join(gla_dir, f'inverted_full_audio' + (str(i)) + '.wav'), inv_signal, sample_rate)
        fig.savefig(os.path.join(gla_dir, f'inverted_full_audio_' + (str(i)) + '.png'))

        i += 1
    '''
    gan_.summarize_models()
    fig = gan_.preview()
    fig.tight_layout()
    plt.show(block=True)
    '''


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
    
    griffin_lim_gan(args.model, args.dataset, args.version, args.samples, args.patch, args.aspect)
