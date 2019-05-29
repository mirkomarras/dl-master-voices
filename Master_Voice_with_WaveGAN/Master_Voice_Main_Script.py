import decimal 
import tensorflow as tf
import os
from keras import backend as K
from keras.models import Model, load_model, Sequential
from keras.layers.core import Flatten, Reshape, Dense
from keras.layers import Input, Dot, Lambda
import librosa
from librosa import display
import logging
import math
import numpy as np
import operator
import pandas as pd
import pickle
import random
from scipy import spatial
from scipy.signal import lfilter
from scipy import signal
from scipy.io.wavfile import write
from IPython.display import display, Audio, HTML
import matplotlib.pyplot as plt
import soundfile as sf
import argparse
import warnings
warnings.filterwarnings('ignore')

from helper import get_fft_spectrum
from helper import load_wav
from helper import framesig
from helper import deframesig
from helper import normalize_frames
from helper import denormalize_frames
from helper import rolling_window
from helper import round_half_up
from helper import save_obj
from helper import load_obj

def run_model(ir_speaker_dir, ir_room_dir, ir_mic_dir,
              batch_size, n_iterations, learning_rate,
              min_similarity, max_similarity, utterance_type,
              post_processing):

    print('')
    print('Current Parameters')
    print('')
    print('Speaker IR directory:', ir_speaker_dir)
    print('Room IR directory:', ir_room_dir)
    print('Mic IR directory:', ir_mic_dir)
    print('Batch size:', batch_size)
    print('Number of iterations:', n_iterations)
    print('Learning rate:', learning_rate)
    print('Min similarity factor:', min_similarity)
    print('Max similairty factor:', max_similarity)
    print('Utterance type:', utterance_type)
    print('Post Processing:', post_processing)
    print('')

    acoustic_params = {'max_sec': 10,
                    'bucket_step': 1,
                    'frame_step': 0.01,
                    'sample_rate': 16000,
                    'preemphasis_alpha': 0.97,
                    'frame_len': 0.025,
                    'num_fft': 512}
                    
    '''
    -- Bottleneck Features Extraction: VGGVox-Vectors
    '''
    print('Importing VGG Vox')
    graph = tf.Graph()
    sess = tf.Session(graph=graph)

    # Creating Keras session
    K.set_session(sess)

    # Import VGG Vox pre-trained model
    with graph.as_default():
        with tf.name_scope('VGG_Vox'):
            bottleneck = load_model('/beegfs/mm10572/voxceleb2/additional_material/vggvox.h5')

    '''
    -- Training Dataset
    '''

    # Pickle File: An array including a list of paths used for training: 50 utterances per person 
    utterance_paths = load_obj('/beegfs/mm10572/voxceleb2/additional_material/train_vox2_abspaths_1000_users') 

    # Pickle File: An array including a list of labels corresponding to paths in utterance_paths
    utterance_labels = load_obj('/beegfs/mm10572/voxceleb2/additional_material/train_vox2_labels_1000_users')

    # Numpy File: A 2D matrix including embedding vectors for paths in utterance_paths
    utterance_bottleneck_features = np.load('/beegfs/mm10572/voxceleb2/additional_material/train_vox2_embs_1000_users.npy') 

    user_id_position = 5
    indexes_male_utterances = []
    indexes_female_utterances = []
    # CSV File: A CSV containing the metadata of the VoxCeleb2 dataset
    vox_metadata = pd.read_csv('/beegfs/mm10572/voxceleb2/additional_material/vox2_meta.csv', header=None, names=['vid', 'vggid', 'gender', 'set']) 
    for p_index, path in enumerate(utterance_paths):
        if (p_index+1) % 100 == 0:
            print('\rPath', p_index+1, '/', len(utterance_paths), end='')
        if vox_metadata.loc[vox_metadata.vid == path.split('/')[user_id_position], 'gender'].values[0] == 'm':
            indexes_male_utterances.append(p_index)
        else:
            indexes_female_utterances.append(p_index)

    '''
    -- Load WaveGAN
    '''

    print('')
    print('Importing WaveGAN')
    # Load WaveGAN graph
    with graph.as_default():
        infer_path = os.path.abspath("/beegfs/kp2218/rename_test/infer/infer.meta")
        print('Path to infer.meta file:',infer_path)
        saver = tf.train.import_meta_graph(infer_path)

        ckpt_path = os.path.abspath("/beegfs/kp2218/rename_test/model.ckpt-62552")
        print('Path to .ckpt file:', ckpt_path)
        saver.restore(sess, ckpt_path)

    '''
    -- Audio Processing Parameters
    '''

    print('Importing Audio Processing Parameters')
    # Speaker IR
    ir_speaker, fs_speaker = librosa.load(ir_speaker_dir, sr=16000, mono=True)

    # Microphone IR
    ir_mic, fs_mic = librosa.load(ir_mic_dir, sr=16000, mono=True)

    # Room IR
    ir_room, fs_room = librosa.load(ir_room_dir, sr=16000, mono=True)

    '''
    Post Processing functions
    '''

    def gan_audio_generator(z):
        
        # WaveGAN audio generator
        G_z = graph.get_tensor_by_name('G_z:0')
        
        return G_z


    def audio_post_processing(G_z):
        
        with tf.name_scope("post_processing") as scope:
            
            '''
            ------- Speaker IR Convolution -------
            '''
                
            # Convolve with Speaker IR
            ir_speaker_tensor = tf.placeholder(dtype=tf.float32, shape=[None], name="Speaker_Tensor")
            ir_speaker_tensor = tf.expand_dims(ir_speaker_tensor, 1)
            ir_speaker_tensor = tf.expand_dims(ir_speaker_tensor, 2)
            
            # Zero pad with speaker IR shape
            pad_len_speaker = int(len(ir_speaker)/2 -1)
            pad_speaker_value = tf.constant([[0,0], [pad_len_speaker,pad_len_speaker+1], [0,0]], name="Speaker_Pad")
            G_z = tf.pad(G_z, pad_speaker_value, "CONSTANT")

            speaker_out = tf.nn.conv1d(G_z, ir_speaker_tensor, 1, padding="VALID", name="Speaker_Out")
            
            '''
            ------- Noise Parameter -------
            '''
            
            # Add noise
            s = speaker_out.get_shape().as_list()
            noise_tensor = tf.random.normal([1,65536,1], mean=0, stddev=5e-3, dtype=tf.float32, name="Noise_param")
            
            # ** Add filter **
            
            speaker_out = tf.add(speaker_out, noise_tensor, name="Speaker_plus_Noise")

            '''
            ------- Room IR Convolution -------
            '''
            
            # Convolve with Room IR
            ir_room_tensor = tf.placeholder(dtype=tf.float32, shape=[None], name="Room_Tensor")
            ir_room_tensor = tf.expand_dims(ir_room_tensor, 1)
            ir_room_tensor = tf.expand_dims(ir_room_tensor, 2)
            
            # Zero pad with Room IR shape
            pad_len_room = int(len(ir_room)/2 -1)
            pad_room_value = tf.constant([[0,0], [pad_len_room,pad_len_room+1], [0,0]])
            speaker_out = tf.pad(speaker_out, pad_room_value, "CONSTANT", name="Room_Pad")
            
            room_out = tf.nn.conv1d(speaker_out, ir_room_tensor, 1, padding="VALID", name="Room_Out")

            '''
            ------- Mic IR Convolution -------
            '''
            
            # Convolve with Mic IR
            ir_mic_tensor = tf.placeholder(dtype=tf.float32, shape=[None], name="Mic_Tensor")
            ir_mic_tensor = tf.expand_dims(ir_mic_tensor, 1)
            ir_mic_tensor = tf.expand_dims(ir_mic_tensor, 2)
            
            # Zero pad with Mic IR shape
            pad_len_mic = int(len(ir_mic)/2 -1)
            pad_mic_value = tf.constant([[0,0], [pad_len_mic,pad_len_mic+1], [0,0]], name="Mic_Pad")
            room_out = tf.pad(room_out, pad_mic_value, "CONSTANT")
            
            audio_out = tf.nn.conv1d(room_out, ir_mic_tensor, 1, padding="VALID", name="Audio_Out")

            return audio_out
        
    def spectrogram_conversion(audio_out):
        
        with tf.name_scope("spectrogram_conv") as scope:
            
            # Converting audio tensor to frames
            frames = tf.contrib.signal.frame(audio_out[0,:,0],
                                            frame_length=int(acoustic_params['frame_len'] * acoustic_params['sample_rate']),
                                            frame_step=int(acoustic_params['frame_step'] * acoustic_params['sample_rate']),
                                            pad_end=True,
                                            name="Frames")
            
            # Windowing each frame
            frames = frames * tf.contrib.signal.hamming_window(int(acoustic_params['frame_len'] * acoustic_params['sample_rate']), periodic=True)

            frames = tf.transpose(frames)

            t = tf.shape(frames)

            # Zero padding for FFT
            pad_amount = tf.zeros([int(acoustic_params['num_fft'])-int(acoustic_params['frame_len'] * acoustic_params['sample_rate']),
                                    t[1]], tf.float32)

            frames_pad = tf.concat([frames, pad_amount], axis=0)

            # Computing the FFT of the audio tensor
            y = tf.cast(frames_pad, tf.complex64)
            y = tf.transpose(y)

            spec = tf.cast(tf.abs(tf.spectral.fft(y, name="FFT")), tf.float32)
            mag_spec = tf.transpose(spec)

            # Normalizing the spectrogram
            mean_tensor, variance_tensor = tf.nn.moments(mag_spec, axes=[1])
            std_tensor = tf.math.sqrt(variance_tensor)

            m_shape = tf.shape(mean_tensor)
            s_shape = tf.shape(std_tensor)

            spec_norm = (mag_spec - tf.reshape(mean_tensor, [m_shape[0],1])) / tf.maximum(tf.reshape(std_tensor, [s_shape[0],1]), 1e-8)

            spec_norm = tf.expand_dims(spec_norm, 0)
            spec_norm = tf.expand_dims(spec_norm, 3)

            return spec_norm

    '''
    -- Adverserial Model (VGG)
    '''
    print('Generating Master Voice Keras model')
    with graph.as_default():
        
        bottleneck_extractor = Model(bottleneck.inputs, Flatten()(bottleneck.output))

        # ----------GAN Layers----------

        # Input Layer
        Z = Input(tensor=graph.get_tensor_by_name('z:0'), name='LatentVector')

        # Layer 1
        Audio_Generation_Using_GAN = Lambda(gan_audio_generator,
                        output_shape=[65536, None],
                        name='WaveGAN')
        gan_audio = Audio_Generation_Using_GAN(Z)

        # Layers 2 and 3
        if post_processing == 'yes':

            Audio_Post_Processing = Lambda(audio_post_processing,
                            output_shape=[65536, 1],
                            name='AudioPostProcessing')
            processed_audio = Audio_Post_Processing(gan_audio)

            Spectrogram_Conversion = Lambda(spectrogram_conversion,
                            output_shape=[512, None, 1],
                            name='Spectrogram01')
            in_a = Spectrogram_Conversion(processed_audio)

        elif post_processing == 'no':

            Spectrogram_Conversion = Lambda(spectrogram_conversion,
                            output_shape=[512, None, 1],
                            name='Spectrogram01')
            in_a = Spectrogram_Conversion(gan_audio)

        # ----------Spectrogram 02 (Dataset)----------

        in_b = Input(shape=(512, None, 1), name='Spectrogram02')

        # ----------Keras Model----------

        inputs = [Z, in_b]

        emb_a = bottleneck_extractor(in_a)
        emb_b = bottleneck_extractor(in_b)
        similarity = Dot(axes=1, normalize=True)([emb_a, emb_b])

        siamese = Model(inputs,similarity)

        # Define input layer of combined model
        if post_processing == 'yes':
            model_input_layer = [siamese.layers[0].input, 
                                graph.get_tensor_by_name('AudioPostProcessing/post_processing/Speaker_Tensor:0'), 
                                graph.get_tensor_by_name('AudioPostProcessing/post_processing/Room_Tensor:0'), 
                                graph.get_tensor_by_name('AudioPostProcessing/post_processing/Mic_Tensor:0'),
                                siamese.layers[4].input]

        elif post_processing == 'no':
            model_input_layer = [siamese.layers[0].input,
                                 siamese.layers[3].input]

        model_output_layer =  siamese.layers[-1].output

        cost_function = model_output_layer[0][0]

        gradient_function = K.gradients(cost_function, Z)[0]

        grab_cost_and_gradients_from_model = K.function(model_input_layer, [cost_function, gradient_function])

        filter_gradients = lambda c, g, t1, t2: [g[i] for i in range(len(c)) if c[i] >= t1 and c[i] <= t2]

    def evaluate_fac(spectrogram, bottleneck_extractor, utterance_paths, utterance_bottleneck_features, threshold):
        
        # Predict speaker using the VGG Vox model
        with graph.as_default():
            bottleneck_features = bottleneck_extractor.predict(spectrogram)[0]
            
        # Compute similairty values
        similarities = [1 - spatial.distance.cosine(bottleneck_features, utterance_bottleneck_features[i]) for i in range(len(utterance_paths))]
        fac = np.sum([1 for s in similarities if s > threshold])
        return fac

    # -------------- Main Loop --------------

    print('')
    print('Training: Start')

    if utterance_type == 'male':
        indexes_optimization = indexes_male_utterances
    elif utterance_type == 'female':
        indexes_optimization = indexes_female_utterances

    # Initliaze latent vector (Z)
    starting_z = (np.random.rand(1, 100) * 2.) - 1
    master_z = starting_z

    # Numpy arrays for storing cost and FAC values
    cost_values = np.zeros(n_iterations)
    fac_values_1 = np.zeros(n_iterations)
    fac_values_2 = np.zeros(n_iterations)

    # Main Loop
    for iteration in range(n_iterations):
            
        costs = []
        gradients = []
            
        for index in random.sample(indexes_optimization, batch_size):

            # Spectrogram of input data
            base_spectrogram, _, _ = get_fft_spectrum(utterance_paths[index], acoustic_params, audio_read='Yes')
            
            # Input into the model   
            if post_processing == 'yes':

                input_pair = ([master_z,
                                ir_speaker,
                                ir_room,
                                ir_mic,
                                np.array([base_spectrogram.reshape(*base_spectrogram.shape, 1)])])

            elif post_processing == 'no':

                input_pair = ([master_z,
                                np.array([base_spectrogram.reshape(*base_spectrogram.shape, 1)])])

                
            # Cost and gradient calculation
            cost, gradient = grab_cost_and_gradients_from_model(input_pair)
            costs.append(np.squeeze(cost))
            gradients.append(np.squeeze(gradient))
            
        filtered_gradients = filter_gradients(costs, gradients, min_similarity, max_similarity)
            
        # Adding the gradients to the latent vector
        perturbation = np.mean(filtered_gradients, axis=0) * learning_rate
        perturbation = np.clip(perturbation, 1e-5, None) # min_change = 1e-5
        master_z += perturbation
        master_z = np.clip(master_z, -1, 1)
            
        # For each iteration, append a cost value to see how it changes over the iterations
        cost_values[iteration] += np.mean(costs)

        # Generating spectrogram of iterated latent vector (z)
        if post_processing == 'yes': 

            iterated_spectrogram = sess.run(siamese.get_layer('Spectrogram01').output, {siamese.get_layer('LatentVector').input: master_z,
                                                                                            graph.get_tensor_by_name('AudioPostProcessing/post_processing/Speaker_Tensor:0'): ir_speaker, 
                                                                                            graph.get_tensor_by_name('AudioPostProcessing/post_processing/Room_Tensor:0'): ir_room, 
                                                                                            graph.get_tensor_by_name('AudioPostProcessing/post_processing/Mic_Tensor:0'): ir_mic})
        
        elif post_processing == 'no':

            iterated_spectrogram = sess.run(siamese.get_layer('Spectrogram01').output, {siamese.get_layer('LatentVector').input: master_z})


        # Determine FAC
        fac_1 = evaluate_fac(iterated_spectrogram, bottleneck_extractor, utterance_paths, utterance_bottleneck_features, threshold=0.53)
        fac_2 = evaluate_fac(iterated_spectrogram, bottleneck_extractor, utterance_paths, utterance_bottleneck_features, threshold=0.74)

        # For each iteration, append the FAC to see how it changes over the iterations
        fac_values_1[iteration] += fac_1
        fac_values_2[iteration] += fac_2

        print('\rStep ' + str(iteration + 1) + '/' + str(n_iterations), '\n',
                '- False Accepts Count (threshold = 0.53)', fac_1, '\n',
                '- False Accepts Count (threshold = 0.74)', fac_2)

    '''
    -- Plots (Cost, FAC, Audio, Spectrogram)
    '''

    plt.figure()

    # Plot average cost values
    plt.subplot(311)
    plt.plot(cost_values)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')

    # Plot FAC
    plt.subplot(312)
    plt.plot(fac_values_1)
    plt.xlabel('Iterations')
    plt.ylabel('FAC (t = 0.53)')

    # Plot FAC
    plt.subplot(313)
    plt.plot(fac_values_2)
    plt.xlabel('Iterations')
    plt.ylabel('FAC (t = 0.74)')

    plt.savefig('Data_plot.png')

    '''
    Audio and Spectrogram (Starting and Master latent vector)
    '''
    # Generate audio and spectrogram of iterated value of _z
    if post_processing == 'yes':

        starting_audio = sess.run(siamese.get_layer('AudioPostProcessing').output, {siamese.get_layer('LatentVector').input: starting_z,
                                                                                        graph.get_tensor_by_name('AudioPostProcessing/post_processing/Speaker_Tensor:0'): ir_speaker, 
                                                                                        graph.get_tensor_by_name('AudioPostProcessing/post_processing/Room_Tensor:0'): ir_room, 
                                                                                        graph.get_tensor_by_name('AudioPostProcessing/post_processing/Mic_Tensor:0'): ir_mic})

        starting_spectrogram = sess.run(siamese.get_layer('Spectrogram01').output, {siamese.get_layer('LatentVector').input: starting_z,
                                                                                        graph.get_tensor_by_name('AudioPostProcessing/post_processing/Speaker_Tensor:0'): ir_speaker, 
                                                                                        graph.get_tensor_by_name('AudioPostProcessing/post_processing/Room_Tensor:0'): ir_room, 
                                                                                        graph.get_tensor_by_name('AudioPostProcessing/post_processing/Mic_Tensor:0'): ir_mic})

        master_audio = sess.run(siamese.get_layer('AudioPostProcessing').output, {siamese.get_layer('LatentVector').input: master_z,
                                                                                        graph.get_tensor_by_name('AudioPostProcessing/post_processing/Speaker_Tensor:0'): ir_speaker, 
                                                                                        graph.get_tensor_by_name('AudioPostProcessing/post_processing/Room_Tensor:0'): ir_room, 
                                                                                        graph.get_tensor_by_name('AudioPostProcessing/post_processing/Mic_Tensor:0'): ir_mic})

        master_spectrogram = sess.run(siamese.get_layer('Spectrogram01').output, {siamese.get_layer('LatentVector').input: master_z,
                                                                                        graph.get_tensor_by_name('AudioPostProcessing/post_processing/Speaker_Tensor:0'): ir_speaker, 
                                                                                        graph.get_tensor_by_name('AudioPostProcessing/post_processing/Room_Tensor:0'): ir_room, 
                                                                                        graph.get_tensor_by_name('AudioPostProcessing/post_processing/Mic_Tensor:0'): ir_mic})

    elif post_processing == 'no':

        starting_audio = sess.run(siamese.get_layer('WaveGAN').output, {siamese.get_layer('LatentVector').input: starting_z})

        starting_spectrogram = sess.run(siamese.get_layer('Spectrogram01').output, {siamese.get_layer('LatentVector').input: starting_z})

        master_audio = sess.run(siamese.get_layer('WaveGAN').output, {siamese.get_layer('LatentVector').input: master_z})

        master_spectrogram = sess.run(siamese.get_layer('Spectrogram01').output, {siamese.get_layer('LatentVector').input: master_z})

    # Display spectrogram
    plt.figure()
    plt.imshow(starting_spectrogram[0,:,:,0])
    plt.title('Spectrogram (Starting _z)')
    plt.savefig('Spec_Starting.png')

    plt.figure()
    plt.imshow(master_spectrogram[0,:,:,0])
    plt.title('Spectrogram (Master _z)')
    plt.savefig('Spec_Master.png')

    # Save audio
    write('Audio_Starting.wav', 16000, starting_audio[0,:,0])
    write('Audio_Master.wav', 16000, master_audio[0,:,0])
    print('Training: End')

def main():
    
    parser = argparse.ArgumentParser(description='Master Voice implementation with WaveGAN')

    # Data Parameters
    parser.add_argument('--batch', dest='batch_size', default=16, type=int, action='store',
                        help='Batch size for training')
    parser.add_argument('--iterations', dest='n_iterations', default=100, type=int, action='store',
                        help='Number of training iterations')
    parser.add_argument('--learning_rate', dest='learning_rate', default=1e-4, type=float, action='store',
                        help='Learning rate')
    parser.add_argument('--min_similarity', dest='min_similarity', default=0.25, type=float, action='store',
                        help='Min similairty metric')
    parser.add_argument('--max_similarity', dest='max_similarity', default=0.75, type=float, action='store',
                        help='Max similairty metric')
    parser.add_argument('--utterance_type', dest='utterance_type', default='male', type=str, action='store',
                        help='Utterance type (male/female)')

    # IR Parameters
    parser.add_argument('--speaker_ir', dest='ir_speaker_dir',
                        default='/beegfs/kp2218/test_runs/conv_test/data/audio/ir_speaker/IR_ClestionBD300.wav',
                        type=str, action='store',
                        help='Speaker IR directory')
    parser.add_argument('--room_ir', dest='ir_room_dir', 
                        default='/beegfs/kp2218/test_runs/conv_test/data/audio/ir_mic/IR_OktavaMD57.wav', 
                        type=str, action='store',
                        help='Room IR directory')
    parser.add_argument('--mic_ir', dest='ir_mic_dir', 
                        default='/beegfs/kp2218/test_runs/conv_test/data/audio/ir_room/BRIR.wav', 
                        type=str, action='store',
                        help='Mic IR directory')

    # Post Processing Parameter
    parser.add_argument('--post_processing', dest='post_processing',
                        default='yes', action='store',
                        help='Post Processing metric')

    args = parser.parse_args()

    run_model(args.ir_speaker_dir, args.ir_room_dir, args.ir_mic_dir,
              args.batch_size, args.n_iterations, args.learning_rate,
              args.min_similarity, args.max_similarity, args.utterance_type,
              args.post_processing)

if __name__ == "__main__":
    main()