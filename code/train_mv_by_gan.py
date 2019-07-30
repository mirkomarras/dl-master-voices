from keras.layers import Input, Dot, Lambda
from keras.models import Model, load_model
from keras.layers.core import Flatten
from helpers.audioutils import *
from helpers.coreutils import *
from keras import backend as K
import tensorflow as tf

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
G_z = None

def evaluate_fac(graph, spectrogram, bottleneck_extractor, utterance_paths, utterance_bottleneck_features, threshold, step=50):
    with graph.as_default():
        bottleneck_features = bottleneck_extractor.predict(spectrogram)[0]
    similarities = [1 - spatial.distance.cosine(bottleneck_features, utterance_bottleneck_features[i]) for i in range(len(utterance_paths))]
    check = [(1 if s > threshold else 0) for s in similarities]
    fac = np.sum(check)
    imp = np.sum([(1 if np.sum(check[x:x + step]) > 0 else 0) for x in range(0, len(check), step)])
    return fac, imp

def spectrogram_conversion(audio_out):
    global args
    with tf.name_scope("spectrogram_conv") as scope:
        frames = tf.contrib.signal.frame(audio_out[0, :, 0], frame_length=int(args.frame_size * args.sample_rate), frame_step=int(args.frame_stride * args.sample_rate), pad_end=True, name="Frames")
        frames = frames * tf.contrib.signal.hamming_window(int(args.frame_size * args.sample_rate), periodic=True)
        frames = tf.transpose(frames)
        t = tf.shape(frames)
        pad_amount = tf.zeros([int(args.num_fft) - int(args.frame_size * args.sample_rate), t[1]], tf.float32)
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
        spec_norm = (mag_spec - tf.reshape(mean_tensor, [m_shape[0], 1])) / tf.maximum(tf.reshape(std_tensor, [s_shape[0], 1]), 1e-8)
        spec_norm = tf.expand_dims(spec_norm, 0)
        spec_norm = tf.expand_dims(spec_norm, 3)
        return spec_norm

def audio_post_processing(G_z):
    global args
    with tf.name_scope("post_processing") as scope:
        ir_speaker_tensor = tf.placeholder(dtype=tf.float32, shape=[None], name="Speaker_Tensor")
        ir_speaker_tensor = tf.expand_dims(ir_speaker_tensor, 1)
        ir_speaker_tensor = tf.expand_dims(ir_speaker_tensor, 2)
        ir_speaker, fs_speaker = librosa.load(args.ir_speaker_dir, sr=args.sample_rate, mono=True)
        ir_mic, fs_mic = librosa.load(args.ir_mic_dir, sr=args.sample_rate, mono=True)
        ir_room, fs_room = librosa.load(args.ir_room_dir, sr=args.sample_rate, mono=True)
        pad_len_speaker = int(len(ir_speaker) / 2 - 1)
        pad_speaker_value = tf.constant([[0, 0], [pad_len_speaker, pad_len_speaker + 1], [0, 0]], name="Speaker_Pad")
        G_z = tf.pad(G_z, pad_speaker_value, "CONSTANT")
        speaker_out = tf.nn.conv1d(G_z, ir_speaker_tensor, 1, padding="VALID", name="Speaker_Out")
        # Noise Parameter
        s = speaker_out.get_shape().as_list()
        noise_tensor = tf.random.normal([1, 65536, 1], mean=0, stddev=5e-3, dtype=tf.float32, name="Noise_param")
        speaker_out = tf.add(speaker_out, noise_tensor, name="Speaker_plus_Noise")
        # Room IR Convolution
        ir_room_tensor = tf.placeholder(dtype=tf.float32, shape=[None], name="Room_Tensor")
        ir_room_tensor = tf.expand_dims(ir_room_tensor, 1)
        ir_room_tensor = tf.expand_dims(ir_room_tensor, 2)
        pad_len_room = int(len(ir_room) / 2 - 1)
        pad_room_value = tf.constant([[0, 0], [pad_len_room, pad_len_room + 1], [0, 0]])
        speaker_out = tf.pad(speaker_out, pad_room_value, "CONSTANT", name="Room_Pad")
        room_out = tf.nn.conv1d(speaker_out, ir_room_tensor, 1, padding="VALID", name="Room_Out")
        # Mic IR Convolution
        ir_mic_tensor = tf.placeholder(dtype=tf.float32, shape=[None], name="Mic_Tensor")
        ir_mic_tensor = tf.expand_dims(ir_mic_tensor, 1)
        ir_mic_tensor = tf.expand_dims(ir_mic_tensor, 2)
        pad_len_mic = int(len(ir_mic) / 2 - 1)
        pad_mic_value = tf.constant([[0, 0], [pad_len_mic, pad_len_mic + 1], [0, 0]], name="Mic_Pad")
        room_out = tf.pad(room_out, pad_mic_value, "CONSTANT")
        audio_out = tf.nn.conv1d(room_out, ir_mic_tensor, 1, padding="VALID", name="Audio_Out")
        return audio_out

def gan_audio_generator(z):
    global graph
    G_z = graph.get_tensor_by_name('G_z:0')
    return G_z

def run_model(noises):
    global args
    global graph

    graph = tf.Graph()
    sess = tf.Session(graph=graph)
    K.set_session(sess)

    # Bottleneck Features Extraction
    print('Importing bottleneck feature model')
    with graph.as_default():
        with tf.name_scope('VGG_Vox'):
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

    # WaveGAN model
    print('Importing WaveGAN model')
    with graph.as_default():
        infer_path = os.path.abspath(args.gan_metafile_path)
        print('Path to infer.meta file:', infer_path)
        saver = tf.train.import_meta_graph(infer_path)
        ckpt_path = args.gan_ckpt_path
        print('Path to .ckpt file:', ckpt_path)
        ckpt_path = os.path.abspath(ckpt_path)
        saver.restore(sess, ckpt_path)

    # Post-processing parameters
    print('Importing post-processing data')
    ir_speaker, fs_speaker = librosa.load(args.ir_speaker_dir, sr=args.sample_rate, mono=True)
    ir_mic, fs_mic = librosa.load(args.ir_mic_dir, sr=args.sample_rate, mono=True)
    ir_room, fs_room = librosa.load(args.ir_room_dir, sr=args.sample_rate, mono=True)

    # Adverserial Model
    print('Generating Master Voice Keras model')
    with graph.as_default():
        bottleneck_extractor = Model(bottleneck.inputs, Flatten()(bottleneck.output))
        Z = Input(tensor=graph.get_tensor_by_name('z:0'), name='LatentVector')
        gan_audio = Lambda(gan_audio_generator, output_shape=[65536, None], name='WaveGAN')(Z)
        if args.post_processing:
            processed_audio = Lambda(audio_post_processing, output_shape=[65536, 1], name='AudioPostProcessing')(gan_audio)
            in_a = Lambda(spectrogram_conversion, output_shape=[512, None, 1], name='Spectrogram01')(processed_audio)
        else:
            in_a = Lambda(spectrogram_conversion, output_shape=[512, None, 1], name='Spectrogram01')(gan_audio)

        in_b = Input(shape=(512, None, 1), name='Spectrogram02')
        inputs = [Z, in_b]

        emb_a = bottleneck_extractor(in_a)
        emb_b = bottleneck_extractor(in_b)
        similarity = Dot(axes=1, normalize=True)([emb_a, emb_b])
        siamese = Model(inputs, similarity)

        if args.post_processing:
            model_input_layer = [siamese.layers[0].input, graph.get_tensor_by_name('AudioPostProcessing/post_processing/Speaker_Tensor:0'), graph.get_tensor_by_name('AudioPostProcessing/post_processing/Room_Tensor:0'), graph.get_tensor_by_name('AudioPostProcessing/post_processing/Mic_Tensor:0'), siamese.layers[4].input]
        else:
            model_input_layer = [siamese.layers[0].input, siamese.layers[3].input]

        model_output_layer = siamese.layers[-1].output
        cost_function = model_output_layer[0][0]
        gradient_function = K.gradients(cost_function, Z)[0]
        grab_cost_and_gradients_from_model = K.function(model_input_layer, [cost_function, gradient_function])
        filter_gradients = lambda c, g, t1, t2: [g[i] for i in range(len(c)) if c[i] >= t1 and c[i] <= t2]


    # Master voice optimization
    print('Starting optimization')

    if args.utterance_type == 'male':
        indexes_optimization = indexes_male_utterances
    else:
        indexes_optimization = indexes_female_utterances

    for attempt in range(args.attempts):
        starting_z = (np.random.rand(1, 100) * 2.) - 1
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

                base_spectrogram = get_fft_spectrum(utterance_paths[batch_sample], args.sample_rate, args.nfilt, noises, args.num_fft, args.frame_size, args.frame_stride, args.preemphasis, args.vad, args.aug, args.prefilter, args.normalize)
                sp_times.append(time.time() - start_time)
                if args.post_processing:
                    input_pair = ([master_z, ir_speaker, ir_room, ir_mic, np.array([base_spectrogram])])
                else:
                    input_pair = ([master_z, np.array([base_spectrogram])])

                # Similarity and gradient calculation
                start_time = time.time()
                cost, gradient = grab_cost_and_gradients_from_model(input_pair)
                gp_times.append(time.time() - start_time)
                costs.append(np.squeeze(cost))
                gradients.append(np.squeeze(gradient))

            filtered_gradients = filter_gradients(costs, gradients, args.min_similarity, args.max_similarity)

            # Adding the gradients to the latent vector
            perturbation = np.mean(filtered_gradients, axis=0) * args.learning_rate
            master_z += perturbation

            # For each iteration, append a cost value to see how it changes over the iterations
            cost_values[iteration] += np.mean(costs)

            # Generating spectrogram of iterated latent vector (z)
            if args.post_processing:
                iterated_spectrogram = sess.run(siamese.get_layer('Spectrogram01').output, {siamese.get_layer('LatentVector').input: master_z, graph.get_tensor_by_name('AudioPostProcessing/post_processing/Speaker_Tensor:0'): ir_speaker, graph.get_tensor_by_name('AudioPostProcessing/post_processing/Room_Tensor:0'): ir_room, graph.get_tensor_by_name('AudioPostProcessing/post_processing/Mic_Tensor:0'): ir_mic})
            else:
                iterated_spectrogram = sess.run(siamese.get_layer('Spectrogram01').output, {siamese.get_layer('LatentVector').input: master_z})

            # Determine FAC
            fac_1, imp_1 = evaluate_fac(graph, iterated_spectrogram, bottleneck_extractor, utterance_paths, utterance_bottleneck_features, threshold=args.thr_eer)
            fac_2, imp_2 = evaluate_fac(graph, iterated_spectrogram, bottleneck_extractor, utterance_paths, utterance_bottleneck_features, threshold=args.thr_far1)

            # For each iteration, append the FAC to see how it changes over the iterations
            fac_values_1[iteration] += fac_1
            fac_values_2[iteration] += fac_2

            text = ''
            if fac_1 > best_fac_1:
                best_fac_1 = fac_1
                best_latent_vector = np.copy(master_z)

                # Generate audio and spectrogram of iterated value of _z
                if args.post_processing:
                    master_audio = sess.run(siamese.get_layer('AudioPostProcessing').output, {siamese.get_layer('LatentVector').input: master_z, graph.get_tensor_by_name('AudioPostProcessing/post_processing/Speaker_Tensor:0'): ir_speaker, graph.get_tensor_by_name('AudioPostProcessing/post_processing/Room_Tensor:0'): ir_room, graph.get_tensor_by_name('AudioPostProcessing/post_processing/Mic_Tensor:0'): ir_mic})
                    starting_audio = sess.run(siamese.get_layer('AudioPostProcessing').output,{siamese.get_layer('LatentVector').input: starting_z,graph.get_tensor_by_name('AudioPostProcessing/post_processing/Speaker_Tensor:0'): ir_speaker,graph.get_tensor_by_name('AudioPostProcessing/post_processing/Room_Tensor:0'): ir_room,graph.get_tensor_by_name('AudioPostProcessing/post_processing/Mic_Tensor:0'): ir_mic})
                else:
                    master_audio = sess.run(siamese.get_layer('WaveGAN').output, {siamese.get_layer('LatentVector').input: master_z})
                    starting_audio = sess.run(siamese.get_layer('WaveGAN').output, {siamese.get_layer('LatentVector').input: starting_z})

                write(os.path.join(args.gan_mv_base_path, args.gan_mv_base_placeholder + str(args.utterance_type) + '_' + str(attempt) + '.wav'), args.sample_rate, master_audio[0, :, 0])
                write(os.path.join(args.gan_mv_base_path, args.gan_ov_base_placeholder + str(args.utterance_type) + '_' + str(attempt) + '.wav'), args.sample_rate, starting_audio[0, :, 0])

                text = '[SAVED]'

            print('Attempt', attempt, '\tSPT', round(np.mean(sp_times), 2), '\tGPT', round(np.mean(gp_times), 2), '\tStep ' + str(iteration + 1) + '/' + str(args.n_iterations), '\t', 'False Accepts (THE@EER):', fac_1, '\t', 'False Accepts (THR@FAR1)', fac_2, '\t', 'Imp (THR@EER):', imp_1, '\t', 'Imp (THR@FAR1)', imp_2, text)

        master_z = np.copy(best_latent_vector)

        # Generate audio and spectrogram of iterated value of _z
        if args.post_processing:
            master_audio = sess.run(siamese.get_layer('AudioPostProcessing').output, {siamese.get_layer('LatentVector').input: master_z, graph.get_tensor_by_name('AudioPostProcessing/post_processing/Speaker_Tensor:0'): ir_speaker, graph.get_tensor_by_name('AudioPostProcessing/post_processing/Room_Tensor:0'): ir_room, graph.get_tensor_by_name('AudioPostProcessing/post_processing/Mic_Tensor:0'): ir_mic})
            starting_audio = sess.run(siamese.get_layer('AudioPostProcessing').output,{siamese.get_layer('LatentVector').input: starting_z, graph.get_tensor_by_name('AudioPostProcessing/post_processing/Speaker_Tensor:0'): ir_speaker,graph.get_tensor_by_name('AudioPostProcessing/post_processing/Room_Tensor:0'): ir_room,graph.get_tensor_by_name( 'AudioPostProcessing/post_processing/Mic_Tensor:0'): ir_mic})
        else:
            master_audio = sess.run(siamese.get_layer('WaveGAN').output,{siamese.get_layer('LatentVector').input: master_z})
            starting_audio = sess.run(siamese.get_layer('WaveGAN').output, {siamese.get_layer('LatentVector').input: starting_z})

        # Save audio
        write(os.path.join(args.gan_mv_base_path, args.gan_mv_base_placeholder + str(args.utterance_type) + '_' + str(attempt) + '.wav'), args.sample_rate, master_audio[0, :, 0])
        write(os.path.join(args.gan_mv_base_path, args.gan_ov_base_placeholder + str(args.utterance_type) + '_' + str(attempt) + '.wav'), args.sample_rate,starting_audio[0, :, 0])

    print('Ending optimization')

def main():
    global args

    parser = argparse.ArgumentParser(description='WaveGAN Master Voice Optimization')

    # Training Parameters
    parser.add_argument('--verifier', dest='verifier', default='', type=str, action='store', help='Type of verifier [xvector|vggvox|resnet34vox|resnet50vox].')
    parser.add_argument('--model_dir', dest='bottleneck_model_path', default='', type=str, action='store', help='Output directory for the trained model')
    parser.add_argument('--noises_dir', dest='noises_dir', default='', type=str, action='store', help='Input noise directory for augmentation')
    parser.add_argument('--speaker_ir', dest='ir_speaker_dir', default='../noise/ir_speaker/IR_ClestionBD300.wav', type=str, action='store', help='Speaker IR file')
    parser.add_argument('--room_ir', dest='ir_room_dir', default='../noise/ir_room/BRIR.wav', type=str, action='store', help='Room IR file')
    parser.add_argument('--mic_ir', dest='ir_mic_dir', default='../noise/ir_mic/IR_OktavaMD57.wav', type=str, action='store', help='Mic IR file')
    parser.add_argument('--post_processing', dest='post_processing', default=True, action='store', help='Post processing flag')

    parser.add_argument('--meta_file', dest='metadata_path', default='', type=str, action='store', help='Dataset metadata')
    parser.add_argument('--meta_gender_col', dest='metadata_gender_column', default='gender', action='store', help='Metadata gender column')
    parser.add_argument('--meta_gender_male', dest='metadata_gender_male', default='m', action='store', help='Metadata gender male')
    parser.add_argument('--train_paths', dest='utterances_path', default='', type=str, action='store', help='MV train paths')
    parser.add_argument('--train_labels', dest='train_labels', default='', type=str, action='store', help='MV train labels')
    parser.add_argument('--train_embs', dest='utterances_features_path', default='', type=str, action='store', help='MV train emeddings')

    parser.add_argument('--gan_metafile_path', dest='gan_metafile_path', default='/beegfs/kp2218/rename_test/infer/infer.meta', action='store', help='GAN metafile path')
    parser.add_argument('--gan_ckpt_path', dest='gan_ckpt_path', default="/beegfs/kp2218/rename_test/model.ckpt-62552", action='store', help='GAN ckpt path')
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
    python ./code/train_mv_by_gan.py
    --verifier "vggvox"
    --model_dir "./models/vggvox/ks_model/vggvox.h5"
    --noises_dir "./data/noise"
    --speaker_ir "./data/noise/ir_speaker/IR_ClestionBD300.wav"
    --room_ir "./data/noise/ir_room/BRIR.wav"
    --mic_ir "./data/noise/ir_mic/IR_OktavaMD57.wav"
    --post_processing True
    --meta_file "./data/vox2_meta/meta_vox2.csv"
    --meta_gender_col "gender"
    --meta_gender_male "m"
    --train_paths "./data/vox2_mv/train_vox2_abspaths_1000_users"
    --train_labels "./data/vox2_mv/train_vox2_labels_1000_users"
    --train_embs "./data/vox2_mv/train_vox2_embs_1000_users.npy"
    --gan_metafile_path "./models/stdgan/infer.meta"
    --gan_ckpt_path "./models/stdgan/model.ckpt-62552"
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