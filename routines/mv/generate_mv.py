'''
 1. Input a GAN
 2. Generate N audio samples
 3. Create a folder in /data/vs_mv/data
 4. Save the audio samples under appropriate folder

'''

import os
import argparse

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from helpers.audio import *
from helpers import plotting

from models import gan

def generate_mv(model, dataset, version, n=32, patch=256, aspect=1, sample_rate=16000):
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

	#setup the output folder
	label = (dataset.split("-")[-1][0]) + "-" + (dataset.split("-")[-1][0])
	out_folder = "spectrogram_" + label + "_sv"
	gla_dir = os.path.join("./data/vs_mv_data/", out_folder, f'v'+(f'{version:03}'))
	if not os.path.exists(gla_dir):
		os.makedirs(gla_dir)

	#use griffin-lim algorithm
	i = 0
	for s in samples:
		print("SAMPLE #{}/{}".format(i+1,len(samples)))

		s2 = s.squeeze()

		#mirror the spectrogram
		sp = np.vstack((s2, np.zeros((1, s2.shape[1])), s2[:0:-1]))

		# clip the audio to remove noise
		sp = sp.clip(0)
		
		#INVERTING
		inv_signal = spectrum_to_signal(sp.T, int(2.57*16000))

		#export the audio
		sf.write(os.path.join(gla_dir, f'gen_' + dataset + f'_0' + (str(i)) + '.wav'), inv_signal, sample_rate)
		

		i += 1


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
	
	generate_mv(args.model, args.dataset, args.version, args.samples, args.patch, args.aspect)
