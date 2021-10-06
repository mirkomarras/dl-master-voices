#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import loguru
from numpy.core.fromnumeric import reshape
import tensorflow as tf
import soundfile as sf
import numpy as np
import os

from datetime import datetime

from models import cloning
from models.mv.attacks import PGDSpectrumDistortion, PGDWaveformDistortion, NESVoiceCloning
from models.verifier.model import Model
from helpers.dataset import Dataset
from helpers import plotting, audio

from loguru import logger

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class SiameseModel(object):
    """
       Class to represent Master Voice (MV) models with master voice training and testing functionalities
    """

    def __init__(self, dir, params, playback=False, ir_dir=None, sample_rate=16000, run_id=None):
        """
        Method to initialize a master voice model that will save audio samples in 'data/vs_mv_data/{net}-{netv}_{gan}-{ganv}_{f|m}-{f|m}_{mv|sv}'
        :param sample_rate:     Sample rate of an audio sample to be processed
        :param dir:          Path to the folder where master voice audio samples will be saved (data/vs_mv_data/vggvox-v004_real_u-f/)
        """
        assert sample_rate > 0, 'Please provide a non-negative sample rate'

        self.gan = None
        self.verifier = None
        self.sample_rate = sample_rate
        self.dir = dir
        self.params = params

        self.playback = playback
        if self.playback:
            assert os.path.isdir(ir_dir), 'Playback simulation is enabled, but impulse response directory is not set {ir_dir}'
            logger.info(f'Setting up playback simulation ({ir_dir})')
            # Load noise data
            self.noise_paths = audio.load_noise_paths(ir_dir)
            self.noise_cache = audio.cache_noise_data(self.noise_paths)
        else:
            self.noise_paths = None
            self.noise_cache = None

        # Create sub-directories for saving seed and master voices
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

        assert os.path.exists(self.dir), 'Please check folder permission for seed and master voice saving'

        # Retrieve the version of the seed and master voices sets for that particular combination of verifier and gan
        if run_id is None:
            self.id = '{:03d}'.format(len(os.listdir(self.dir)))
        else:
            self.id = '{:03d}'.format(run_id)

        # Create sub-directories for saving seed and master voices
        if not os.path.exists(self.dir_full):
            logger.debug(f'Created output dir: {self.dir_full}')
            os.makedirs(self.dir_full)
        else:
            logger.debug(f'Output dir exists: {self.dir_full}')

        self.save_params()

        assert os.path.exists(self.dir_full) and os.path.exists(self.dir_full), 'Please check folder permission for seed and master voice version saving'

    def save_params():
        with open(os.path.join(self.dir_full, 'params.txt'), 'w') as file:
            file.write(json.dumps(self.params))

    def setup_attack(self, attack_type):
        if attack_type == 'nes@cloning':
            self.attack = NESVoiceCloning(self.siamese_model, text='The assistant is triggered by saying hey google', n=10, sigma=0.1, antithetic=True)
        elif attack_type == 'pgd@spec':
            self.attack = PGDSpectrumDistortion(self.siamese_model)
        elif attack_type == 'pgd@wave':
            self.attack = PGDWaveformDistortion(self.siamese_model, self.playback, self.noise_paths, self.noise_cache)
        else:
            raise ValueError(f'Attack not implemented: {attack_type}')

    @property
    def dir_full(self):
        return os.path.join(self.dir, 'v' + self.id)


    def set_generator(self, gan):
        """
        Method that load and set the GAN model that will generate fake audio/spectrogram samples
        :param gan:     GAN model
        """
        assert gan is not None, 'The GAN passed as input is None'

        gan.load()

        self.gan = gan


    def set_verifier(self, verifier):
        """
        Method to build, load, and set the verifier that will be used for master voice optimization
        :param verifier:    Verifier model
        """
        assert verifier is not None, 'The verifier passed as input is None'

        # Build and load the model
        verifier.build(mode='test')
        verifier.load()
        
        # Run evaluation and calibrate decision thresholds 
        verifier.calibrate_thresholds()

        self.verifier = verifier


    def build(self, fft_size=256):
        """
        Method to create a siamese model, starting from the current verifier one branch generates fake gan samples, the other branch received real audio samples
        """

        # We set up the left branch of the siamese model (to be used for feeding training spectrograms)
        signal_input = tf.keras.Input(shape=(fft_size, None, 1,))
        embedding_1 = self.verifier.infer()(signal_input)

        if self.gan is not None:
            # We set up the right branch of the siamese model (to be used for feeding mv spectrograms generated by the gan)
            another_signal_input = self.gan.get_generator().input
            normalized_spectrum_input = audio.tf_normalize_frames(self.gan.get_generator().output[-1])
            embedding_2 = self.verifier.infer()(normalized_spectrum_input)
        else:
            # We set up the right branch of the siamese model (to be used for feeding the spectrogram of the current seed voice)
            another_signal_input = tf.keras.Input(shape=(fft_size, None, 1))
            embedding_2 = self.verifier.infer()(another_signal_input)

        # We set up a layer to compute the cosine similarity between the speaker embeddings
        similarity = tf.keras.layers.Dot(axes=1, normalize=True)([embedding_1, embedding_2])

        # We create a model that, given two input examples, returns the cosine similarity between the corresponding embeddings
        self.siamese_model = tf.keras.Model([signal_input, another_signal_input], similarity)


    def defaults(self):

        class AttrDict(object):

            def __init__(self, specs):
                """ Creates a new parameter specification.
                :param specs: a dict in the following format {'<param-name>': tuple(<default>, <type, <validator>)}
                """
                self.__dict__['_specs'] = specs

            def add(self, specs):
                self._specs.update(specs)

            def __getattr__(self, name):
                if name in self._specs:
                    return self._specs[name]
                else:
                    raise KeyError(name)

            def __setattr__(self, key, value):
                raise ValueError('Values cannot be set directly. Use the `update` method.')

            def update(self, specs):
                self._specs.update(specs)

        return AttrDict({
            'gradient': 'normed',
            'l2_regularization': 0,
            'learning_rate': 1e-1,
            'n_epochs': 5,
            'patience': 3,
            'max_attack_vector': 0,
            'min_sim': 0
        })


    def batch_optimize_by_path(self, seed_voice, train_data, test_gallery, settings=None):
        """
        Batch master voice optimization from seed utterances given by file/dir names.

        :param seed_voice:          Seed voice to
        :param train_data:          Real audio data against which master voices are optimized - shape (None,1)
        :param test_data:           Real user against which the current master voice are validated

        """

        # Extract the speaker embeddings for validating the impersonation rate of the current master voice:
        # - x_mv_test is a list of audio files for validation (ten consecutive audio files per user)
        # - y_mv_test is a list of user labels corresponding to each audio file in x_mv_test
        # - male_x_mv_test is a list of user labels in y_mv_test associated to male users
        # - female_x_mv_test is a list of user labels in y_mv_test associated to female users
        #
        # The list x_mv_test_embs will include the embeddings corresponding to the audio files in x_mv_test
        # x_mv_test, y_mv_test, male_x_mv_test, female_x_mv_test = test_data

        extractor = self.verifier.infer()

        seed_voices = [seed_voice] if not os.path.isdir(seed_voice) else [os.path.join(seed_voice, voice) for voice in sorted(os.listdir(seed_voice))]
        n_seed_voices = len(seed_voices) if self.gan is None else int(seed_voices)

        # Find the last computed sample
        existing_files = [x for x in os.listdir(self.dir_full) if x.startswith('opt_progress')]
        n_seed_start = len(existing_files)

        if n_seed_start > 0:
            with open(os.path.join(self.dir_full, 'stats.json'), 'r') as f:
                stats = json.load(f)
        else:
            # Collect stats from all seed voices
            stats = {k: [] for k in 'max_dist,l2_norm,mv_eer_results,mv_far1_results,sv_eer_results,sv_far1_results'.split(',')}
        
        for iter in range(n_seed_start, n_seed_voices): # For each audio sample to batch_optimize_by_path
            logger.info(f'Starting optimization {seed_voices[iter]}: {iter+1} of {n_seed_voices}')
            #, , ':', iter+1, 'of', n_seed_voices, '- GAN:', self.gan)

            # We initialize the starting latent vector / spectrogram to batch_optimize_by_path (mv stands for master voice, sv stands for seed voice)
            # if self.gan is not None:
            #     input_mv, input_avg, input_std = (tf.random.normal(size=(128)).astype(np.float32), None, None)
            # else:

            input_sv = audio.decode_audio(seed_voices[iter]).astype(np.float32)
            # Clip to max 3 seconds
            max_length = 3 * 16000
            if input_sv.shape[0] > max_length:
                logger.warning(f'Clipping speech to {max_length} samples')
                input_sv = input_sv[:max_length]
            input_mv, performance = self.optimize(input_sv, train_data, test_gallery, settings)

            gender = self.params.mv_gender[0] # Gender selector: 'm' or 'f'
            model_suffix = '' if self.gan is not None else seed_voices[iter].split('/')[-1].split('.')[0]
            
            # TODO [Mess] This conversion should not be necessary - reconsider params to save method
            # input_sv, input_avg, input_std = audio.get_np_spectrum(input_sv, self.sample_rate, num_fft=512, full=False)
            # input_sv = input_sv[..., np.newaxis]

            # TODO Remove me - added for temporary debugging
            self.test_gallery = test_gallery
            self.save(input_sv, input_mv, performance, model_suffix, test_gallery.pop_file)
            
            logger.info(f'Finished optimization! {gender} impersonation {performance["mv_far1_results"][0][gender]:.3f} -> {performance["mv_far1_results"][-1][gender]:.3f}')

            stats['mv_eer_results'].append(performance['mv_eer_results'][-1][gender].item())
            stats['mv_far1_results'].append(performance['mv_far1_results'][-1][gender].item())
            stats['sv_eer_results'].append(performance['mv_eer_results'][0][gender].item())
            stats['sv_far1_results'].append(performance['mv_far1_results'][0][gender].item())
            stats['l2_norm'].append(performance['l2_norm'][-1])
            stats['max_dist'].append(performance['max_dist'][-1])

            # Summarize all
            with open(os.path.join(self.dir_full, 'stats.json'), 'w') as f:
                json.dump(stats, f, indent=4)

        return stats #os.path.join(self.dir_full, 'stats.json')


    def optimize(self, input_sv, train_data, test_gallery=None, settings=None):
        """

        :param input_sv:
        :param test_gallery:
        :param train_data: a data pipeline yielding speaker embeddings of the training population
        :param settings: optimization settings (dict)
        :return:
        """

        settings = settings or self.defaults()

        metrics = ('mv_eer_results', 'mv_far1_results', 'mv_avg_similarity', 'l2_norm', 'max_dist')
        performance = {m: [] for m in metrics}

        remaining_attempts = settings.patience
        best_value_attempt = 0.0

        # input_sv = self.attack.prep(input_sv)
        input_sv, perturbation = self.attack.setup(input_sv)
        logger.debug(f'Configured optimization: parameter space {perturbation.shape}')
        # perturbation = np.zeros_like(input_sv, dtype=np.float32)
        # perturbation = tf.convert_to_tensor(perturbation, dtype='float32')
        input_mv = tf.convert_to_tensor(input_sv, dtype='float32')
        input_sv = tf.convert_to_tensor(input_sv, dtype='float32')

        # Get baseline stats
        results = self.test(input_mv, test_gallery)
        performance['mv_eer_results'].append(results[0])
        performance['mv_far1_results'].append(results[1])
        performance['l2_norm'].append(tf.reduce_mean(tf.square(perturbation)).numpy().item())
        performance['max_dist'].append(np.max(np.abs(perturbation)).item())

        logger.debug('(Baseline) Imp@EER m={:.3f} f={:.3f} | Imp@FAR1 m={:.3f} f={:.3f}'.format(results[0]["m"], results[0]["f"], results[1]["m"], results[1]["f"]), end='\n')

        for epoch in range(settings.n_epochs):  # For each optimization epoch
            t1 = datetime.now()
            epoch_similarities = []

            perturbation, epoch_similarities = self.attack.optimize(input_sv, perturbation, train_data, settings)

            epoch_loss = tf.reduce_mean(epoch_similarities).numpy().item()

            t2 = datetime.now()
            if test_gallery is not None:
                
                # TODO hard-coded for spectrogram optimization
                input_mv = self.attack.run(input_sv, perturbation)
                # input_mv = input_sv + perturbation
                # input_mv = tf.clip_by_value(input_mv, 0, 10000)

                # We test the current master voice version for impersonation rates on the validation set
                results = self.test(input_mv, test_gallery)
                performance['mv_avg_similarity'].append(epoch_loss)
                performance['mv_eer_results'].append(results[0])
                performance['mv_far1_results'].append(results[1])
                performance['l2_norm'].append(tf.reduce_mean(tf.square(perturbation)).numpy().item())
                performance['max_dist'].append(np.max(np.abs(perturbation)).item())

                if (results[0]['m'] + results[0]['f']) > best_value_attempt:  # Check if the total impersonation rate after the current epoch is improved
                    best_value_attempt = results[0]['m'] + results[0]['f']  # Update the best impersonation rate value
                    remaining_attempts = settings.patience  # Resume remaining attempts to patience times
                    # print(' - Best Score', end='')
                else:
                    remaining_attempts -= 1  # Reduce the remaining attempts to improve the impersonation rate
                    # print(f' - Attempts ({remaining_attempts})', end='')

                if remaining_attempts == 0:  # If there are no longer remaining attempts we start the optimization of the current voice
                    break

            t3 = datetime.now()
            opt_time = (t2 - t1).total_seconds()
            val_time = (t3 - t2).total_seconds()
            logger.debug('(Epoch={:2d}) Imp@EER m={:.3f} f={:.3f} | Imp@FAR1 m={:.3f} f={:.3f} | opt time {:.1f} + val time {:.1f}'.format(
                epoch, results[0]["m"], results[0]["f"], results[1]["m"], results[1]["f"], opt_time, val_time))

        return self.attack.run(input_sv, perturbation), performance

    def save(self, seed_sample, attack_sample, performance_stats, filename='', population_name='default', iter=None): # input_avg, input_std, 
        """
        
        Save the seed and attack samples along with optimization stats and impersonation scores:
        - 

         original and optimized master voices

        :param iter:               current iteration (optional) 
        :param seed_sample:        Seed sample
        :param attack_sample:      Attack sample
        :param performance_stats:  Dict with optimization stats
        """

        # assert input_sv.ndim == 3
        # assert input_sv.shape == input_mv.shape
        suffix = filename if iter is None else f'{filename}_{iter}'
        # suffix = '{}'.format(iter) if self.gan is not None else filename

        # Prepare the seed sample
        seed_spec = self.ensure_spectrogram(seed_sample, numpy=True).squeeze()
        seed_wave = self.ensure_waveform(seed_sample)

        # Prepare the attack sample
        mv_spec = self.ensure_spectrogram(attack_sample, numpy=True).squeeze()
        mv_wave = self.ensure_waveform(attack_sample, aux_signal=seed_wave)

        _, input_avg, input_std = audio.get_np_spectrum(seed_wave.ravel())
        mv_spec_denormed = np.squeeze(np.squeeze(audio.denormalize_frames(np.squeeze(mv_spec), input_avg, input_std)))
        seed_spec_denormed = np.squeeze(np.squeeze(audio.denormalize_frames(np.squeeze(seed_spec), input_avg, input_std)))
        
        # Save spectrograms and wave files
        logger.info(f'Saving speech samples to {self.dir_full}/{{sv/mv}}')

        for xv in ('sv', 'mv'):
            if not os.path.exists(os.path.join(self.dir_full, xv)):
                os.makedirs(os.path.join(self.dir_full, xv))

        np.save(os.path.join(self.dir_full, 'sv', suffix), seed_spec)
        sf.write(os.path.join(self.dir_full, 'sv', suffix + '.wav'), seed_wave, self.sample_rate)
        np.save(os.path.join(self.dir_full, 'mv', suffix), mv_spec)
        sf.write(os.path.join(self.dir_full, 'mv', suffix + '.wav'), mv_wave, self.sample_rate)

        # # We save the current audio associated to the master voice latent vector / spectrogram
        # if self.gan is not None:
        #     sp = self.gan.get_generator()(np.expand_dims(input_mv, axis=0))[-1].numpy()
        #     sp = np.squeeze(sp)
        # else:
        #     sp = np.squeeze(audio.denormalize_frames(np.squeeze(input_mv), input_avg, input_std))
        # sp = np.vstack((sp, np.zeros((1, sp.shape[1])), sp[:0:-1]))
        # sp = sp.clip(0)
        # sp_mv = sp

        # We save the current master voice latent vector, if we are using a GAN-based procedure
        # if self.gan is not None:
        #     np.save(os.path.join(self.dir_full, 'mv_' + suffix), input_mv)

        # We save the current audio associated to the seed voice latent vector / spectrogram
        # sp = np.squeeze(self.gan.get_generator()(np.expand_dims(input_sv, axis=0))[-1].numpy()) if self.gan is not None else np.squeeze(audio.denormalize_frames(np.squeeze(input_sv), input_avg, input_std))
        # sp = np.vstack((sp, np.zeros((1, sp.shape[1])), sp[:0:-1]))
        # sp = sp.clip(0)

        # # We save the current seed voice latent vector, if we are using a GAN-based procedure
        # if self.gan is not None:
        #     np.save(os.path.join(self.dir_full, 'sv_' + str(iter)), input_sv)

        # # We save the unnormalized spectrogram of the seed voice        
        # np.save(os.path.join(self.dir_full, 'sv_' + suffix), sp)

        # inv_signal = audio.spectrum_to_signal(sp.T, int((sp.shape[1] + 1) / 100.0 * self.sample_rate), verbose=False)
        # sf.write(os.path.join(self.dir_full, 'sv_' + suffix + '.wav'), inv_signal, self.sample_rate)

        # Plot a comparion between seed and attack spectrograms -------------------------------------------------------
        gender = self.params.mv_gender[0] # Gender selector: 'm' or 'f'

        ir_start = performance_stats['mv_far1_results'][0][gender]
        ir_end = performance_stats['mv_far1_results'][-1][gender]

        n_bins = seed_spec.shape[0]
        assert n_bins in (256, 512), "Could not recognize the number of FFT bins"
        filename_fig = os.path.join(self.dir_full, 'spectrums_' + suffix + '.png')        
        fig = plotting.imsc(
            (mv_spec_denormed, seed_spec_denormed, np.abs(mv_spec_denormed - seed_spec_denormed)), 
            ['master voice (IR_{}={:.2f}) []'.format(gender, ir_end), 'seed voice (IR_{}={:.2f}) []'.format(gender, ir_start), 'diff []'],
            cmap='jet', ncols=3)
        fig.savefig(filename_fig, bbox_inches='tight')

        # We save the similarities, impostor, and gender impostor results ---------------------------------------------
        net = self.params.netv.replace('/', '_')
        filename_format = f'{net}_{population_name}_{{}}_any_{suffix}.npz'

        # TODO Remove me - added for temporary debugging
        mv_spec_recomp = audio.get_tf_spectrum(mv_wave[tf.newaxis, ...])
        results_ori = self.test(mv_spec[tf.newaxis, ..., tf.newaxis], self.test_gallery)
        results_ref = self.test(seed_spec[tf.newaxis, ..., tf.newaxis], self.test_gallery)
        results_recomp = self.test(mv_spec_recomp, self.test_gallery)

        # TODO Remove me - temporary sanity check
        logger.warning(f'(Seed) Imp@EER {gender}={results_ref[0][gender]} | Imp@FAR1 {gender}={results_ref[1][gender]}')
        logger.warning(f'(Optimized) Imp@EER {gender}={results_ori[0][gender]} | Imp@FAR1 {gender}={results_ori[1][gender]}')
        logger.warning(f'(Inverted) Imp@EER {gender}={results_recomp[0][gender]} | Imp@FAR1 {gender}={results_recomp[1][gender]}')

        performance_stats['imp_seed'] = [results_ref[0][gender], results_ref[1][gender]]
        performance_stats['imp_opt'] = [results_ori[0][gender], results_ori[1][gender]]
        performance_stats['imp_inv'] = [results_recomp[0][gender], results_recomp[1][gender]]

        # During optimization, we keep track of impersonation performance only for the `any` policy?
        for thr in ('eer', 'far1'):
            # TODO [Critical] Why save scores that are pre-populated in some member arrays?
            results = {'sims': self.sims[thr], 'imps': self.imps[thr], 'gnds': self.gnds[thr]}
            filename_stats = os.path.join(self.dir_full, filename_format.format(thr))
            logger.info(f'Saving progress stats to {filename_stats}')
            np.savez(filename_stats, results)

        # We update and save the current impersonation rate history
        filename_stats = os.path.join(self.dir_full, f'opt_progress_{suffix}.npz')
        logger.info(f'Saving progress stats to {filename_stats}')
        np.savez(filename_stats, **performance_stats)

        # TODO [Maybe] Should save stats in plain text also - to make external parsing easier


    def test(self, input_mv, test_gallery):
        """

        Test impersonation rate of a single speech sample in a given population. Return array broken down by gender:

        np.array -> {
            'm': [<impersonation rate> for each threshold],
            'f': [<impersonation rate> for each threshold]
        }

        :param input_mv:            np array with the speech sample (or latent vector)
        :param test_gallery:           dataset with the test population
        :return:
        """

        if test_gallery is None:
            return {'m': np.array(0), 'f': np.array(0)}, {'m': np.array(0), 'f': np.array(0)}

        input_spectrum = self.ensure_spectrogram(input_mv)

        # TODO This should not be populated in the background - need a better way of passing the results around
        self.sims, self.imps, self.gnds = {}, {}, {}

        # tf.expand_dims(input_spectrum, axis=0)
        sim_df, imp_df, gnd_df = self.verifier.test_error_rates(input_spectrum, test_gallery, level='eer')
        eer_results = {'m': np.mean(gnd_df[:, 0]), 'f': np.mean(gnd_df[:, 1])}
        self.sims['eer'] = sim_df
        self.imps['eer'] = imp_df
        self.gnds['eer'] = gnd_df

        sim_df, imp_df, gnd_df = self.verifier.test_error_rates(input_spectrum, test_gallery, level='far1')
        far1_results = {'m': np.mean(gnd_df[:, 0]), 'f': np.mean(gnd_df[:, 1])}
        self.sims['far1'] = sim_df
        self.imps['far1'] = imp_df
        self.gnds['far1'] = gnd_df

        return eer_results, far1_results

    def ensure_spectrogram(self, sample, numpy=False):
        # Ensure spectrogram input
        # 1-dim = (1, 512) and smaller - GAN 
        # 2-dim = (1, 32000) - waveform
        # 3-dim = (1, 256, 400) - spectrum
        # 4-dim = (1, 256, 400, 1) - spectrum
        if sample.ndim == 1 and sample.shape[0] > 512:
            spectrum = audio.get_tf_spectrum(sample[tf.newaxis, ...])
        elif sample.ndim == 2 and sample.shape[-1] > 512:
            spectrum = audio.get_tf_spectrum(sample)
        elif sample.ndim == 2:
            # TODO [Mess] This should ultimately be removed and GAN sampling moved to attack class
            spectrum, _, _ = audio.normalize_frames(np.squeeze(self.gan.get_generator()(np.expand_dims(sample, axis=0))[-1].numpy(), axis=0))
        elif 2 < sample.ndim < 5:
            spectrum = sample
        else:
            raise ValueError(f'Invalid input shape: {sample.shape}')
        
        if hasattr(spectrum, 'numpy') and numpy:
            spectrum = spectrum.numpy()

        return spectrum

    def ensure_waveform(self, sample, aux_signal=None):
        # self.gan.get_generator()(np.expand_dims(input_sv, axis=0))[-1].numpy())
        if hasattr(sample, 'numpy'):
            sample = sample.numpy()

        if hasattr(sample, 'ndim'):
            if sample.ndim == 1:
                return sample.ravel()
            elif sample.ndim == 2 and np.max(sample.shape) < 1025:
                return sample.ravel()
            elif sample.ndim >= 2:

                if aux_signal is None:
                    logger.warning('Signal stats not available - assuming an unnormalized spectrogram')
                    sp = np.squeeze(sample)
                else:
                    _, input_avg, input_std = audio.get_np_spectrum(aux_signal.ravel())
                    sp = np.squeeze(np.squeeze(audio.denormalize_frames(np.squeeze(sample), input_avg, input_std)))

                sp = np.vstack((sp, np.zeros((1, sp.shape[1])), sp[:0:-1]))
                sp = sp.clip(0)

                inv_signal = audio.spectrum_to_signal(sp.T, int((sp.shape[1] + 1) / 100.0 * self.sample_rate), verbose=False)
                return inv_signal

        elif isinstance(sample, tuple):
            # A tuple of spectrogram, average stats, std stats
            # inv_signal = audio.spectrum_to_signal(sp.T, int((sp.shape[1] + 1) / 100.0 * self.sample_rate), verbose=False)
            input_sv, input_avg, input_std = sample

            sp = np.squeeze(np.squeeze(audio.denormalize_frames(np.squeeze(input_sv), input_avg, input_std)))
            sp = np.vstack((sp, np.zeros((1, sp.shape[1])), sp[:0:-1]))
            sp = sp.clip(0)

            inv_signal = audio.spectrum_to_signal(sp.T, int((sp.shape[1] + 1) / 100.0 * self.sample_rate), verbose=False)
            return inv_signal