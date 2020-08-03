# combination of create_pairs.py and test_pairs.py

from scipy.spatial.distance import cosine
import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import os

from helpers.audio import load_noise_paths, cache_noise_data, decode_audio, get_tf_spectrum, get_tf_filterbanks

from models.verifier.thinresnet34 import ThinResNet34
from models.verifier.resnet50 import ResNet50
from models.verifier.resnet34 import ResNet34
from models.verifier.xvector import XVector
from models.verifier.vggvox import VggVox

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main():
	parser = argparse.ArgumentParser(description='Tensorflow master voice trial pairs generation and testing')

	# ---- CREATION ARGUMENTS ---- #

	# Parameters for verifier
	parser.add_argument('--mv_set', dest='mv_set', default='', type=str, action='store', help='Folder associated to the target master voice set')
	parser.add_argument('--mv_trials', dest='mv_trials', default='./data/vs_mv_pairs/trial_pairs_vox2_mv.csv', type=str, action='store', help='Trials base path for master voice analysis waveforms')

	# ---- TESTER ARGUMENTS ---- #

	 # Parameters for verifier
	parser.add_argument('--net', dest='net', default='', type=str, action='store', help='Network model architecture')

	# Parameters for master voice analysis
	parser.add_argument('--mv_base_path', dest='mv_base_path', default='./data', type=str, action='store', help='Trials base path for master voice analysis waveforms')
	parser.add_argument('--test_list', dest='test_list', default='./data/vs_mv_pairs/mv', type=str, action='store', help='Master voice population to be tested')
	parser.add_argument('--save_path', dest='save_path', default='./data/pt_models/', type=str, help='Path for model and logs')

	# Parameters for raw audio
	parser.add_argument('--sample_rate', dest='sample_rate', default=16000, type=int, action='store', help='Sample rate audio')
	parser.add_argument('--n_seconds', dest='n_seconds', default=3, type=int, action='store', help='Segment lenght in seconds')
	parser.add_argument('--noise_dir', dest='noise_dir', default='./data/vs_noise_data', type=str, action='store', help='Noise directory')

	parser.add_argument('--avg_ver', dest='avg_ver', action='store_true', help='Execute average 10 verification policy')
	parser.set_defaults(avg_ver=False)

	args = parser.parse_args()

	if(args.mv_set == ''):
		print("ERROR: Please specify audio sample directory for '--mv_set'!")
		print("   ex. --mv_set 'spectrogram_m-m_sv/v001'")
		exit()

	if(args.net == ''):
		print("ERROR: Please specify model network for '--net'!")
		print("   ex. --net 'resnet50/v000'")
		exit()


	print('Parameters summary')
	print('>', '-- PAIR CREATION -- ')
	print('>', 'Master voice set: {}'.format(args.mv_set))
	print('>', 'Master voice trials: {}'.format(args.mv_trials))
	print('')

	print('>', '-- PAIR TESTING -- ')
	output_type = ('filterbank' if args.net.split('/')[0] == 'xvector' else 'spectrum')

	print('>', 'Net: {}'.format(args.net))
	print('>', 'Mode: {}'.format(output_type))

	print('>', 'Master voice base path: {}'.format(args.mv_base_path))
	print('>', 'Test list: {}'.format(args.test_list))
	print('>', 'Save path: {}'.format(args.save_path))

	print('>', 'Sample rate: {}'.format(args.sample_rate))
	print('>', 'Maximum number of seconds: {}'.format(args.n_seconds))
	print('>', 'Noise dir: {}'.format(args.noise_dir))
	if args.avg_ver:
		print('>', 'Running AVG 10 Verification policy')


	# -- preloading -- #

	# Load noise data
	print('Load impulse response paths')
	noise_paths = load_noise_paths(args.noise_dir)
	print('Cache impulse response data')
	noise_cache = cache_noise_data(noise_paths, sample_rate=args.sample_rate)

	# Create and restore model
	print('Creating model')
	available_nets = {'xvector': XVector, 'vggvox': VggVox, 'resnet50': ResNet50, 'resnet34': ResNet34, 'thin_resnet': ThinResNet34}
	model = available_nets[args.net.split('/')[0]](id=int(args.net.split('/')[1].replace('v','')))
	model.build(classes=0, mode='test')
	model.load()

	# Create save path
	result_save_path = os.path.join(args.save_path, args.net, 'mvcmp_any')
	if not os.path.exists(result_save_path):
		os.makedirs(result_save_path)

	# -- creation of pair csv file -- #
	for mv_set in ([args.mv_set] if args.mv_set else [os.path.join(mv_set, version) for mv_set in os.listdir('./data/vs_mv_data') for version in os.listdir(os.path.join('./data/vs_mv_data', mv_set))]):
		for mv_file in os.listdir(os.path.join('./data/vs_mv_data', mv_set)):
			print('> processing set', mv_set, 'File', mv_file)
			df = pd.read_csv(args.mv_trials, names=['label', 'path1', 'gender'])
			df['path2'] = os.path.join('vs_mv_data', mv_set, mv_file)

			# Create save path
			if not os.path.exists(os.path.join('data', 'vs_mv_pairs', 'mv', mv_set)):
				os.makedirs(os.path.join('data', 'vs_mv_pairs', 'mv', mv_set))

			df[['label', 'path1', 'path2', 'gender']].to_csv(os.path.join('data', 'vs_mv_pairs', 'mv', mv_set, mv_file.replace('.wav', '.csv')), index=False, header=False)
			print('> saved', mv_file, 'scores in', os.path.join('data', 'vs_mv_pairs', 'mv', mv_set, mv_file.replace('.wav', '.csv')))


	# -- load and test -- #
	# Compute similarity scores
	print('Compute similarity scores')
	extractor = model.infer()
	embs = {}
	if os.path.isdir(args.test_list):
		for mvset in os.listdir(args.test_list):
			for version in os.listdir(os.path.join(args.test_list, mvset)):
				for tfile in os.listdir(os.path.join(args.test_list, mvset, version)):
					if not os.path.exists(os.path.join(result_save_path, mvset, version, tfile)):
						print('> opening', os.path.join(args.test_list, mvset, version, tfile))
						df_1 = pd.read_csv(os.path.join(args.test_list, mvset, version, tfile), names=['label', 'path1', 'path2', 'gender'])

						fp_tfile = os.path.join(args.test_list, mvset, version, tfile)
						sc = []
						lab = []


						emb10p1 = []		#10 embedding samples per user test to use for averaging
						avg10_sc = []
						avg10_lab = []

						for index, row in df_1.iterrows():
							if row['path1'] in embs:
								emb_1 = embs[row['path1']]
							else:
								audio_1 = decode_audio(os.path.join(args.mv_base_path, row['path1']))
								audio_1 = audio_1.reshape((1, -1, 1))
								inp_1 = get_tf_spectrum(audio_1) if output_type == 'spectrum' else get_tf_filterbanks(audio_1)
								emb_1 = tf.keras.layers.Lambda(lambda emb1: tf.keras.backend.l2_normalize(emb1, 1))(extractor.predict(inp_1))
								embs[row['path1']] = emb_1

							if row['path2'] in embs:
								emb_2 = embs[row['path2']]
							else:
								audio_2 = decode_audio(os.path.join(args.mv_base_path, row['path2']))
								audio_2 = audio_2.reshape((1, -1, 1))
								inp_2 = get_tf_spectrum(audio_2) if output_type == 'spectrum' else get_tf_filterbanks(audio_2)
								emb_2 = tf.keras.layers.Lambda(lambda emb2: tf.keras.backend.l2_normalize(emb2, 1))(extractor.predict(inp_2))
								embs[row['path2']] = emb_2

							computed_score = 1 - cosine(emb_1, emb_2)

							lab.append(row['label'])
							sc.append(computed_score)

							#add for avg 10 policy done later
							emb10.append(embs[row['path1']])

							# after 10 samples
							if (index + 1) % 10 == 0:
								print('\r> pair', index + 1, '/', len(df_1.index), '-', mvset, version, tfile, '- ver score example', computed_score, end='')


								#perform average 10 policy
								if args.avg_ver:
									#get average of samples and master voice embeddings to compare
									avg_emb = np.average(emb10, axis=0)  #create average embedding of the 10
									master_emb = embs[row['path2']]      #use most recent path2 assuming the last 10 paths for the master voice are the same
									
									#compute cosine similarity
									avg_cos_score = 1 - cosine(avg_emb, master_emb)
									
									#save to set like with any policy above
									avg10_lab.append(row['label'])
									avg10_sc.append(avg_cos_score)


								emb10 = []		#reset the avg 10 embedding list (even if not in use)



						print()

						df = pd.DataFrame(list(zip(sc, lab)), columns=['score', 'label'])
						df['path1'] = df_1['path1']
						df['path2'] = df_1['path2']
						df['gender'] = df_1['gender']

						if not os.path.exists(os.path.join(result_save_path, mvset, version, 'any')):
							os.makedirs(os.path.join(result_save_path, mvset, version, 'any'))
						df.to_csv(os.path.join(result_save_path, mvset, version, 'any', tfile), index=False)

						print('> saved', fp_tfile, 'scores in', os.path.join(result_save_path, mvset, version, tfile))

						#save average 10 policy output
						if args.avg_ver:
							#set up dataframe to export to csv
							df10 = pd.DataFrame(list(zip(avg10_sc, avg10_lab)), columns=['score', 'label'])
							df10['master_voice'] = df_1['path2']
							df10['gender'] = df_1['gender']

							#check if directory exists first
							output_avg10_dir = os.path.join(result_save_path, mvset, version, 'avg10')
							if not os.path.exists(output_avg10_dir):
								os.makedirs(output_avg10_dir)
							
							#export to csv
							df10.to_csv(os.path.join(output_avg10_dir, tfile), index=False)

					else:
						print('> skipped', os.path.join(result_save_path, mvset, version, tfile))

if __name__ == '__main__':
	main()

