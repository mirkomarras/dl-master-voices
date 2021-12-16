#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.metrics import roc_curve
from loguru import logger
import pandas as pd
import numpy as np
import argparse
import warnings
import sys
import os
import tensorflow as tf
import json
from collections import defaultdict

from helpers.dataset import get_mv_analysis_users, load_data_set, filter_by_gender, load_mv_data, Dataset
from helpers.datapipeline import data_pipeline_mv
from helpers import utils

from models.mv.model import SiameseModel
from models import verifier
from models import gan

utils.setup_logging(level='DEBUG')

logger.debug('sys.path:', sys.path)

warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main():
    parser = argparse.ArgumentParser(description='Master voice training')


    # Parameters for verifier
    group = parser.add_argument_group('Basics')
    group.add_argument('--netv', dest='netv', default='vggvox/v000', type=str, action='store', help='Speaker verification model, e.g., xvector/v000')
    group.add_argument('--attack', dest='attack', default='pgd@spec', type=str, action='store', help='Attack type: pgd@spec, pgd@wave, nes@cloning')
    group.add_argument('--seed', dest='seed_voice', default='data/vs_mv_seed/female/ori_00.wav', type=str, action='store', help='Path to the seed sample(s)')
    group.add_argument('--gender', dest='mv_gender', default='female', type=str, choices=['neutral', 'male', 'female'], action='store', help='Geneder against which master voices will be optimized')

    # Parameters for generative adversarial model or of the seed voice (if netg is specified, the master voices will be created by sampling spectrums from the GAN; otherwise, you need to specify a seed voice to batch_optimize_by_path as a master voice)
    group.add_argument('--gm', dest='gm', default=None, type=str, action='store', help='Generative model, e.g., ms-gan/v000')
    # group.add_argument('--netg_gender', dest='netg_gender', default='neutral', type=str, choices=['neutral', 'male', 'female'], action='store', help='Training gender of the generative adversarial model')

    # Parameters for master voice optimization
    group = parser.add_argument_group('Population')
    group.add_argument('--audio_dir', dest='audio_dir', default='./data/voxceleb2/dev', type=str, action='store', help='Path to the folder where master voice training audios are stored')
    group.add_argument('--audio_meta', dest='audio_meta', default='./data/vs_mv_pairs/meta_data_vox12_all.csv', type=str, action='store', help='Path to the CSV file with id-gender metadata of master voice training audios')
    group.add_argument('--dataset', dest='dataset', default='dev-test', type=str, action='store', help='JSON file with population settings (quick setup)')
    group.add_argument('--train_pop', dest='train_pop', default='./data/vs_mv_pairs/mv_train_population_debug_20u_20s.csv', type=str, action='store', help='Path to the filename-user_id pairs for mv training')
    group.add_argument('--test_pop', dest='test_pop', default='./data/vs_mv_pairs/mv_test_population_debug_20u_10s.csv', type=str, action='store', help='Path to the filename-user_id pairs for mv testing')

    group = parser.add_argument_group('Optimization Settings')
    # group.add_argument('--n_examples', dest='n_examples', default=100, type=int, action='store', help='Number of master voices sampled to be created (only if netg is set)')
    group.add_argument('--gradient', dest='gradient', default=None, action='store', help='Gradient mode: none / pgd / normalized')
    group.add_argument('--n_steps', dest='n_steps', default=3, type=int, action='store', help='Number of optimization steps (passes over the population)')
    group.add_argument('--epsilon', dest='epsilon', default=0.01, type=float, action='store', help='Allowed distortion budget (L_inf)')
    group.add_argument('--step_size', dest='step_size_override', default=None, type=float, action='store', help='Manually override step size')

    group.add_argument('--max_dist', dest='max_dist', default=0, type=float, action='store', help='Max distortion (L∞)')
    group.add_argument('--l2_reg', dest='l2_reg', default=0, type=float, action='store', help='Distortion penalty (L2 regularization)')
    group.add_argument('--clip_av', dest='clip_av', default=0, type=float, action='store', help='Distortion limit (L_inf constraint) ')
    group.add_argument('--lambda_reg', dest='lambda_reg', default=1, type=float, action='store', help='Regularization parameter')
    group.add_argument('--batch', dest='batch', default=16, type=int, action='store', help='Batch size for the optimization')

    group = parser.add_argument_group('Misc')
    group.add_argument('--n_templates', dest='n_templates', default=10, type=int, action='store', help='Number of enrolment templates per user (used for testing impersonation)')
    group.add_argument('--sample_rate', dest='sample_rate', default=16000, type=int, action='store', help='Audio sampling rate')
    group.add_argument('--n_seconds', dest='n_seconds', default=2.58, type=float, action='store', help='Length in seconds of an audio for master voice optimization')
    group.add_argument('--prefetch', dest='prefetch', default=200, type=int, action='store', help='Number of samples to prefetch')
   
    group.add_argument('--run_id', dest='run_id', default=None, type=int, action='store', help='Run ID if you need to resume (defaults to None which creates a new ID each time)')
    group.add_argument('--memory-growth', dest='memory_growth', action='store_true', help='Enable dynamic memory growth in Tensorflow')                       

    group = parser.add_argument_group('Playback Simulation')
    group.add_argument('--play', dest='playback', default=False, action='store_true', help='Simulate playback at optimization time')
    group.add_argument('--ir_dir', dest='ir_dir', default='./data/vs_noise_data/', type=str, action='store', help='Path to the folder with impuse responses (room, micropone, speaker)')
    group.add_argument('-impulse_flags','--impulse_flags', nargs='+', help='Impulse Flags for controlling playback', required=False)
    args = parser.parse_args()

    # 
    supported_attacks = 'pgd@spec,pgd@wave,nes@cloning,pgd@vae,nes@wave'.split(',')
    if args.attack not in supported_attacks:
        raise ValueError('Unsupported attack vector: {args.attack}')

    output_type = ('filterbank' if args.netv.split('/')[0] == 'xvector' else 'spectrum')

    assert '/v' in args.netv, 'The speaker verification model should be given as <model>/v<version>, e.g., vggvox/v003'

    if args.memory_growth:
        physical_devices = tf.config.list_physical_devices("GPU")
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if args.dataset:
        with open('config/datasets.json') as f:
            data_config = defaultdict()
            data_config.update(json.load(f)[args.dataset])
        args.train_pop = data_config['train']
        args.test_pop = data_config['test']
        args.audio_dir = data_config['dir']
        args.audio_meta = data_config['meta']

    # Parameter summary to print at the beginning of the script

    # arg_groups={}

    # for group in parser._action_groups:
    #     group_dict={a.dest:getattr(args,a.dest,None) for a in group._group_actions}
    #     arg_groups[group.title]=argparse.Namespace(**group_dict)

    display_fields = {'netv', 'attack', 'seed_voice', 'mv_gender', 'n_steps'}

    if len(tf.config.get_visible_devices("GPU")) == 0:
        logger.warning(f'No GPU? TF Devices: {tf.config.get_visible_devices()}')
    else:
        logger.info(f'GPU Found: {tf.config.get_visible_devices("GPU")}')

    logger.info('Parameters summary:')
    for key, value in vars(args).items():
        if key in display_fields:
            logger.info('  {:>15s}: {}'.format(key, value))

    train_set = pd.read_csv(args.train_pop)
    x_train, y_train = train_set['filename'], train_set['user_id']
    x_train, y_train = filter_by_gender(x_train, y_train, args.audio_meta, args.mv_gender)
    assert len(x_train) > 0, 'Looks like no user data was loaded! Check your data directories and gender filters'

    logger.info('Initializing siamese network for master voice optimization')
    # Output will be saved in sub-dirs in ./data/vs_mv_data - e.g., 'vggvox-v003_real_f_sv'
    dir_name = utils.sanitize_path(f'{args.netv}_{args.attack}_{args.mv_gender[0]}'.replace('/', '_'))

    # We initialize the siamese model that will be used to batch_optimize_by_path master voices
    siamese_model = SiameseModel(dir=os.path.join('data', 'vs_mv_data', dir_name), params=args, playback=args.playback, ir_dir=args.ir_dir, sample_rate=args.sample_rate, run_id=args.run_id, impulse_flags=args.impulse_flags)
    logger.info('Siamese network initialized')

    logger.info('Setting verifier')
    # We initialize, build, and load a pre-trained speaker verification model; this model will be duplicated in order to create the siamese model
    sv = verifier.get_model(args.netv)
    siamese_model.set_verifier(sv)

    # if args.netg is not None:
    #     logger.info('Setting generator')
    #     # If we want to create mv from GAN examples, we initialize, build, and load a pre-trained GAN into the siamese model
    #     siamese_model.set_generator(gan.get_model(args.netg, args.netg_gender))

    logger.info('Building siamese model')
    # We build a siamese model by duplicating the selected speaker verifier; hence, the siamese model will take two spectrograms as input and return the cosine similarity
    # between the speaker embeddings associated to those spectrograms. If netg is set, the right branch of the model will be fed with spectrograms generated by the GAN.
    siamese_model.build()

    # Sanity check for the speaker encoder
    e = sv.predict(x_train[0])
    logger.debug(f'Speaker embedding sanity check: shape={e.shape}, min={np.min(e):.3f}, max={np.max(e):.3f}')

    logger.info('Checking data pipeline output (print every 100th batch)')
    # We check the output of the master voice optimization pipeline, i.e., spectrograms extracted from the training audio files and their associated user labels
    train_data = data_pipeline_mv(x_train, y_train, int(args.sample_rate * args.n_seconds), args.sample_rate, args.batch, args.prefetch, output_type)

    for index, x in enumerate(train_data):
        if index % 100 == 0:
            logger.debug(f'  {index} -> {x[0].shape}, {x[1].shape}')

    # Setup training and testing datasets
    train_data = data_pipeline_mv(x_train, y_train, int(args.sample_rate * args.n_seconds), args.sample_rate, args.batch, args.prefetch, output_type)

    # Create the testing gallery
    logger.info('Setting up enrolled population & generating speaker embeddings...')
    test_gallery = Dataset(args.test_pop)
    test_gallery.precomputed_embeddings(sv)
    assert test_gallery.population.shape[0] == test_gallery.embeddings.shape[0], "Number of fetched embeddings does not match #people in the population. Outdated cache?"

    # Construct optimization args
    opt_settings = siamese_model.defaults()
    opt_settings.update({
        'gradient': args.gradient,
        'n_steps': args.n_steps,
        'epsilon': args.epsilon,
        'step_size_override': args.step_size_override, 
        'l2_regularization': args.l2_reg,
        'clip_av': args.clip_av,
        'patience': 3
    })

    if opt_settings.step_size_override:
        logger.warning(f'Optimization configured to use `step_size_override` ({opt_settings.step_size_override})!')

    logger.info(f'Running baseline error evaluation [SV thresholds={sv._thresholds}]')

    # Sanity check for error rates
    sanity_samples = ('data/vs_mv_seed/female/', 'data/vs_mv_seed/male/')
    filenames = [os.path.join(sanity_samples[0], file) for file in os.listdir(sanity_samples[0]) if file.endswith('.wav')]
    filenames += [os.path.join(sanity_samples[1], file) for file in os.listdir(sanity_samples[1]) if file.endswith('.wav')]
    embeddings = sv.predict(np.array(filenames))

    sim_matrix, imp_matrix, gnd_matrix = sv.test_error_rates(embeddings, test_gallery, policy='avg', level='far1')

    imp_rates = imp_matrix.sum(axis=1) / len(np.unique(test_gallery.user_ids))
    
    logger.warning(f'Impersonation rate sanity check [avg,far1]: {100 * np.mean(imp_rates):.1f}%')
    logger.debug(f'Gender breakown [m,f]: {np.mean(100 * gnd_matrix, 0).round(2)}')

    # Run optimization
    siamese_model.setup_attack(args.attack, args.gm) # pgd@spec, nes@cloning, pgd@wave
    siamese_model.batch_optimize_by_path(args.seed_voice, train_data, test_gallery, settings=opt_settings)

if __name__ == '__main__':
    main()
