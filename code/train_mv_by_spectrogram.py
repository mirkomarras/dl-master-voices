from keras.layers import Input, Dot, Lambda
from keras.models import Model, load_model
from keras.layers.core import Flatten
from helpers.audioutils import *
from helpers.coreutils import *
from keras import backend as K
import tensorflow as tf

import soundfile as sf
from scipy.io.wavfile import write
from scipy import spatial

import pandas as pd
import numpy as np
import argparse
import warnings
import librosa
import random
import time
import os

warnings.filterwarnings('ignore')

args = None

def stft_for_reconstruction(x):
    global args
    frames = framesig(x, frame_len=args.frame_size * args.sample_rate, frame_step=args.frame_stride * args.sample_rate, winfunc=np.hamming)
    fft_norm = np.fft.fft(frames, n=args.num_fft)
    return fft_norm

def istft_for_reconstruction(X, total_lenght):
    global args
    frames = np.fft.ifft(X, n=args.num_fft)
    x = deframesig(frames, total_lenght, frame_len=args.num_fft, frame_step=args.frame_stride * args.sample_rate, winfunc=np.hamming)
    return x

def reconstruct_signal_griffin_lim(total_lenght, fft, iterations):
    x_reconstruct = np.random.randn(total_lenght)
    reconstruction_spectrogram = stft_for_reconstruction(x_reconstruct)
    while fft.shape[0] != reconstruction_spectrogram.shape[0]:
        total_lenght -= 1
        x_reconstruct = np.random.randn(total_lenght)
        reconstruction_spectrogram = stft_for_reconstruction(x_reconstruct)
    n = iterations
    while n > 0:
        n -= 1
        reconstruction_spectrogram = stft_for_reconstruction(x_reconstruct)
        reconstruction_angle = np.angle(reconstruction_spectrogram)
        proposal_spectrogram = fft * np.exp(1.0j * reconstruction_angle)
        prev_x = x_reconstruct
        x_reconstruct = istft_for_reconstruction(proposal_spectrogram, total_lenght)
        diff = np.sqrt(np.sum((fft - abs(proposal_spectrogram))**2)/fft.size)
        print('\rReconstruction iteration: {}/{} RMSE: {} '.format(iterations - n, iterations, diff), end='')
    print()
    return x_reconstruct

def spectrogram_invertion(master_voice_path, starting_spectrogram, starting_waveform_mean, starting_waveform_std, iterations_griffin_lim=300):
    global args
    starting_waveform, starting_rate = librosa.load(master_voice_path, sr=args.sample_rate, mono=True)
    denormalized_starting_spectrogram = denormalize_frames(starting_spectrogram, starting_waveform_mean, starting_waveform_std)
    return reconstruct_signal_griffin_lim(len(starting_waveform), denormalized_starting_spectrogram.T, iterations=iterations_griffin_lim)

def evaluate_fac(spectrogram, bottleneck_extractor, utterance_paths, utterance_bottleneck_features, threshold, step=50):
    bottleneck_features = bottleneck_extractor.predict(np.array([spectrogram.reshape(spectrogram.shape[0], spectrogram.shape[1], 1)]))[0]
    similarities = [1 - spatial.distance.cosine(bottleneck_features, utterance_bottleneck_features[i]) for i in range(len(utterance_paths))]
    check = [(1 if s > threshold else 0) for s in similarities]
    fac = np.sum(check)
    imp = np.sum([(1 if np.sum(check[x:x + step]) > 0 else 0) for x in range(0, len(check), step)])
    return fac, imp

def run_model(noises):
    global args

    # Bottleneck Features Extraction
    print('Importing bottleneck feature model')
    bottleneck = load_model(args.bottleneck_model_path)

    # Training Data
    print('Importing training data')
    utterance_paths = load_obj(args.utterances_path)
    utterance_bottleneck_features = np.load(args.utterances_features_path)
    user_id_position, indexes_male_utterances, indexes_female_utterances = 5, [], []
    vox_metadata = pd.read_csv(args.metadata_path, header=None, names=['vid', 'vggid', 'gender', 'set'])
    for p_index, path in enumerate(utterance_paths):
        if vox_metadata.loc[vox_metadata.vid == path.split('/')[user_id_position], args.metadata_gender_column].values[0] == args.metadata_gender_male:
            indexes_male_utterances.append(p_index)
        else:
            indexes_female_utterances.append(p_index)

    # Adverserial Model
    print('Generating Master Voice Keras model')
    bottleneck_extractor = Model(bottleneck.inputs, Flatten()(bottleneck.output))
    in_a = Input(shape=(512, None, 1), name='Spectrogram01')
    in_b = Input(shape=(512, None, 1), name='Spectrogram02')
    inputs = [in_a, in_b]
    emb_a = bottleneck_extractor(in_a)
    emb_b = bottleneck_extractor(in_b)
    similarity = Dot(axes=1, normalize=True)([emb_a, emb_b])
    siamese = Model(inputs, similarity)
    model_input_layer = [siamese.layers[0].input, siamese.layers[1].input]
    model_output_layer = siamese.layers[-1].output
    cost_function = model_output_layer[0][0]
    gradient_function = K.gradients(cost_function, model_input_layer)[0]
    grab_cost_and_gradients_from_model = K.function(model_input_layer, [cost_function, gradient_function])
    filter_gradients = lambda c, g, t1, t2: [g[i] for i in range(len(c)) if c[i] >= t1 and c[i] <= t2]

    # Master voice optimization
    print('Starting optimization')

    if args.utterance_type == 'male':
        indexes_optimization = indexes_male_utterances
    else:
        indexes_optimization = indexes_female_utterances

    for attempt in range(args.attempts):
        starting_z, starting_waveform_mean, starting_waveform_std = get_fft_spectrum(args.seed, args.sample_rate, args.nfilt, noises, args.num_fft, args.frame_size, args.frame_stride, args.preemphasis, args.vad, args.aug, args.prefilter, args.normalize)
        master_z = np.copy(starting_z)

        cost_values = np.zeros(args.n_iterations)
        fac_values_1 = np.zeros(args.n_iterations)
        fac_values_2 = np.zeros(args.n_iterations)

        best_fac_1 = 0
        best_latent_vector = None

        for iteration in range(args.n_iterations):

            costs = []
            gradients = []
            batch_samples = random.sample(indexes_optimization, args.batch_size)

            sp_times = []
            gp_times = []
            for index, batch_sample in enumerate(batch_samples):
                start_time = time.time()
                base_spectrogram, _, _ = get_fft_spectrum(utterance_paths[batch_sample], args.sample_rate, args.nfilt, noises, args.num_fft, args.frame_size, args.frame_stride, args.preemphasis, args.vad, args.aug, args.prefilter, args.normalize)
                sp_times.append(time.time() - start_time)
                input_pair = ([np.array([master_z.reshape(master_z.shape[0], master_z.shape[1], 1)]), np.array([base_spectrogram.reshape(base_spectrogram.shape[0], base_spectrogram.shape[1], 1)])])
                # Similarity and gradient calculation
                start_time = time.time()
                cost, gradient = grab_cost_and_gradients_from_model(input_pair)
                gp_times.append(time.time() - start_time)
                costs.append(np.squeeze(cost))
                gradients.append(np.squeeze(gradient))

            filtered_gradients = filter_gradients(costs, gradients, args.min_similarity, args.max_similarity)

            # Adding the gradients to the latent vector
            perturbation = np.mean(filtered_gradients, axis=0) * args.learning_rate
            perturbation = perturbation.reshape(perturbation.shape[0], perturbation.shape[1])
            master_z += perturbation

            # For each iteration, append a cost value to see how it changes over the iterations
            cost_values[iteration] += np.mean(costs)

            # Generating spectrogram of iterated latent vector (z)
            iterated_spectrogram = master_z

            # Determine FAC
            fac_1, imp_1 = evaluate_fac(iterated_spectrogram, bottleneck_extractor, utterance_paths, utterance_bottleneck_features, threshold=args.thr_eer)
            fac_2, imp_2 = evaluate_fac(iterated_spectrogram, bottleneck_extractor, utterance_paths, utterance_bottleneck_features, threshold=args.thr_far1)

            # For each iteration, append the FAC to see how it changes over the iterations
            fac_values_1[iteration] += fac_1
            fac_values_2[iteration] += fac_2

            text = ''
            if fac_1 > best_fac_1:
                best_fac_1 = fac_1
                best_latent_vector = np.copy(master_z)
                to_save_original = spectrogram_invertion(args.seed, starting_z, starting_waveform_mean, starting_waveform_std)
                to_save_master = spectrogram_invertion(args.seed, master_z, starting_waveform_mean, starting_waveform_std)
                sf.write(os.path.join(args.gan_mv_base_path, args.gan_mv_base_placeholder + str(args.utterance_type) + '_' + str(attempt) + '.wav'), to_save_original, args.sample_rate)
                sf.write(os.path.join(args.gan_mv_base_path, args.gan_ov_base_placeholder + str(args.utterance_type) + '_' + str(attempt) + '.wav'), to_save_master, args.sample_rate)

                text = '[SAVED]'

            print('Attempt', attempt, '\tSPT', round(np.mean(sp_times), 2), '\tGPT', round(np.mean(gp_times), 2), '\tStep ' + str(iteration + 1) + '/' + str(args.n_iterations), '\t', 'False Accepts (THE@EER):', fac_1, '\t', 'False Accepts (THR@FAR1)', fac_2, '\t', 'Imp (THR@EER):', imp_1, '\t', 'Imp (THR@FAR1)', imp_2, text)

        master_z = np.copy(best_latent_vector)

        # Save audio
        to_save_original = spectrogram_invertion(args.seed, starting_z, starting_waveform_mean, starting_waveform_std)
        to_save_master = spectrogram_invertion(args.seed, master_z, starting_waveform_mean, starting_waveform_std)
        sf.write(os.path.join(args.gan_mv_base_path, args.gan_mv_base_placeholder + str(args.utterance_type) + '_' + str(attempt) + '.wav'), to_save_original, args.sample_rate)
        sf.write(os.path.join(args.gan_mv_base_path, args.gan_ov_base_placeholder + str(args.utterance_type) + '_' + str(attempt) + '.wav'), to_save_master, args.sample_rate)

    print('Ending optimization')

def main():
    global args

    parser = argparse.ArgumentParser(description='WaveGAN Master Voice Optimization')

    # Training Parameters
    parser.add_argument('--verifier', dest='verifier', default='', type=str, action='store', help='Type of verifier [xvector|vggvox|resnet34vox|resnet50vox].')
    parser.add_argument('--seed', dest='seed', default='', type=str, action='store', help='Utterance to be optimized')
    parser.add_argument('--model_dir', dest='bottleneck_model_path', default='', type=str, action='store', help='Output directory for the trained model')
    parser.add_argument('--noises_dir', dest='noises_dir', default='', type=str, action='store', help='Input noise directory for augmentation')
    parser.add_argument('--post_processing', dest='post_processing', default=True, action='store', help='Post processing flag')

    parser.add_argument('--meta_file', dest='metadata_path', default='', type=str, action='store', help='Dataset metadata')
    parser.add_argument('--meta_gender_col', dest='metadata_gender_column', default='gender', action='store', help='Metadata gender column')
    parser.add_argument('--meta_gender_male', dest='metadata_gender_male', default='m', action='store', help='Metadata gender male')
    parser.add_argument('--train_paths', dest='utterances_path', default='', type=str, action='store', help='MV train paths')
    parser.add_argument('--train_labels', dest='train_labels', default='', type=str, action='store', help='MV train labels')
    parser.add_argument('--train_embs', dest='utterances_features_path', default='', type=str, action='store', help='MV train emeddings')

    parser.add_argument('--gan_mv_base_path', dest='gan_mv_base_path', default='/beegfs/mm10572/master-voices/code_ext/sets/unset', action='store', help='GAN mv base path')
    parser.add_argument('--gan_ov_base_placeholder', dest='gan_ov_base_placeholder', default='audio_original_', action='store', help='GAN mv base placeholder')
    parser.add_argument('--gan_mv_base_placeholder', dest='gan_mv_base_placeholder', default='audio_master_', action='store', help='GAN mv base placeholder')

    parser.add_argument('--batch_size', dest='batch_size', default=128, type=int, action='store', help='Batch size for training')
    parser.add_argument('--n_iterations', dest='n_iterations', default=1000, type=int, action='store', help='Number of training iterations')
    parser.add_argument('--learning_rate', dest='learning_rate', default=1e-2, type=float, action='store', help='Learning rate')
    parser.add_argument('--min_similarity', dest='min_similarity', default=0.25, type=float, action='store', help='Min similairty metric')
    parser.add_argument('--max_similarity', dest='max_similarity', default=1.00, type=float, action='store', help='Max similairty metric')
    parser.add_argument('--utterance_type', dest='utterance_type', default='female', type=str, action='store', help='Utterance type (male/female)')
    parser.add_argument('--attempts', dest='attempts', default=50, type=int, action='store', help='Number of attempts')

    parser.add_argument('--params_file_path', dest='params_file_path', default='params.txt', action='store', help='Parameters file path')
    parser.add_argument('--thr_eer', dest='thr_eer', default=0.53, type=float, action='store', help='Parameters file path')
    parser.add_argument('--thr_far1', dest='thr_far1', default=0.74, type=float, action='store', help='Parameters file path')

    # Acoustic parameters
    parser.add_argument('--sample_rate', dest='sample_rate', default=16000, type=int, action='store', help='Audio sample rate.')
    parser.add_argument('--preemphasis', dest='preemphasis', default=0.97, type=float, action='store', help='Pre-emphasis alpha')
    parser.add_argument('--frame_stride', dest='frame_stride', default=0.01, type=float, action='store', help='Frame stride')
    parser.add_argument('--frame_size', dest='frame_size', default=0.025, type=float, action='store', help='Frame size')
    parser.add_argument('--num_fft', dest='num_fft', default=512, type=int, action='store', help='Number of FFT values')
    parser.add_argument('--min_chunk_size', dest='min_chunk_size', default=10, type=int, action='store', help='Minimum x-axis size of a filterbank')
    parser.add_argument('--max_chunk_size', dest='max_chunk_size', default=300, type=int, action='store', help='Maximum x-axis size of a filterbank')
    parser.add_argument('--aug', dest='aug', default=0, type=int, action='store', help='Augmentation mode [0:no aug, 1:aug any, 2:aug seq, 3:aug_prob]')
    parser.add_argument('--vad', dest='vad', default=False, type=bool, action='store', help='Voice activity detection mode')
    parser.add_argument('--prefilter', dest='prefilter', default=True, type=bool, action='store', help='Prefilter mode')
    parser.add_argument('--normalize', dest='normalize', default=True, type=bool, action='store', help='Normalization mode')
    parser.add_argument('--nfilt', dest='nfilt', default=24, type=int, action='store', help='Number of filterbanks')

    args = parser.parse_args()

    noises = {}
    for type in os.listdir(args.noises_dir):
        noises[type] = []
        for file in os.listdir(os.path.join(args.noises_dir, type)):
            noises[type].append(os.path.join(args.noises_dir, type, file))

    print('Starting WaveGAN Master Voice Optimization')
    if not os.path.exists(args.gan_mv_base_path):
        os.makedirs(args.gan_mv_base_path)
    with open(os.path.join(args.gan_mv_base_path, args.params_file_path), "w") as text_file:
        text_file.write(str(args))

    run_model(noises)

if __name__ == "__main__":
    main()


    '''
    python ./code/train_mv_by_spectrogram.py
    --verifier "vggvox"
    --seed "/beegfs/mm10572/voxceleb1/test/id10293/X7uOKQUYTCM/00006.wav"
    --model_dir "./models/vggvox/ks_model/vggvox.h5"
    --noises_dir "./data/noise"
    --post_processing True
    --meta_file "./data/vox2_meta/meta_vox2.csv"
    --meta_gender_col "gender"
    --meta_gender_male "m"
    --train_paths "./data/vox2_mv/train_vox2_abspaths_1000_users"
    --train_labels "./data/vox2_mv/train_vox2_labels_1000_users"
    --train_embs "./data/vox2_mv/train_vox2_embs_1000_users.npy"
    --gan_mv_base_path "./sets/sample-set"
    --gan_ov_base_placeholder "audio_original_"
    --gan_mv_base_placeholder "gan_mv_base_placeholder"
    --batch_size 128
    --n_iterations 1000
    --learning_rate 1e-2
    --min_similarity 0.25
    --max_similarity 1.00
    --utterance_type "female"
    --attempts 50
    --params_file_path "params_file_path"
    --thr_eer 0.53
    --thr_far1 0.74
    --sample_rate 16000
    --preemphasis 0.97
    --frame_stride 0.01
    --frame_size 0.025
    --num_fft 512
    --min_chunk_size 10
    --max_chunk_size 300
    --aug 0
    --vad False
    --prefilter True
    --normalize True
    --nfilt 24
'''