#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import argparse
import json
import os

from helpers.audio import load_noise_paths, cache_noise_data, decode_audio
from helpers.dataset import load_test_data_from_file, load_mv_data, create_template_trials

from models.verifier.resnet50vox import ResNet50Vox
from models.verifier.resnet34vox import ResNet34Vox
from models.verifier.xvector import XVector
from models.verifier.vggvox import VggVox

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main():
    parser = argparse.ArgumentParser(description='Master voice testing')

    # Parameters for verifier
    parser.add_argument('--net', dest='net', default='', type=str, action='store', help='Network model architecture')
    parser.add_argument('--policy', dest='policy', default='any', type=str, action='store', help='Verification policy')
    parser.add_argument('--n_templates', dest='n_templates', default=1, type=int, action='store', help='Number of enrolment templates')
    parser.add_argument('--n_attacks', dest='n_attacks', default=1, type=int, action='store', help='Number of joint attacks')

    # Parameters for testing verifier against eer
    parser.add_argument('--sv_base_path', dest='sv_base_path', default='./data/vs_voxceleb1/test', type=str, action='store', help='Trials base path for computing speaker verification thresholds')
    parser.add_argument('--sv_pair_path', dest='sv_pair_path', default='./data/ad_voxceleb12/vox1_trial_pairs.csv', type=str, action='store', help='CSV file label, path_1, path_2 speaker verification trials')
    parser.add_argument('--sv_n_pair', dest='sv_n_pair', default=1000, type=int, action='store', help='Number of speaker verification trials')

    # Parameters for master voice analysis
    parser.add_argument('--mv_base_path', dest='mv_base_path', default='./data/vs_voxceleb2/', type=str, action='store', help='Trials base path for master voice analysis waveforms')
    parser.add_argument('--mv_meta', dest='mv_meta', default='./data/ad_voxceleb12/vox2_mv_data.npz', type=str, action='store', help='Numpy data for master voice analysis metadata')
    parser.add_argument('--audio_meta', dest='audio_meta', default='./data/ad_voxceleb12/vox12_meta_data.csv', type=str, action='store', help='CSV file with id-gender metadata')
    parser.add_argument('--n_templates', dest='n_templates', type=int, default=10, action='store', help='Enrolment set size')

    # Parameters for master voice population to be tested
    parser.add_argument('--mv_set', dest='mv_set', default='', type=str, action='store', help='Master voice population to be tested')

    # Parameters for raw audio
    parser.add_argument('--sample_rate', dest='sample_rate', default=16000, type=int, action='store', help='Sample rate audio')
    parser.add_argument('--n_seconds', dest='n_seconds', default=3, type=int, action='store', help='Segment lenght in seconds')
    parser.add_argument('--noise_dir', dest='noise_dir', default='./data/vs_noise_data', type=str, action='store', help='Noise directory')
    parser.add_argument('--augment', dest='augment', default=0, type=int, choices=[0,1], action='store', help='Data augmentation mode')

    args = parser.parse_args()

    print('Parameters summary')

    print('>', 'Net: {}'.format(args.net))
    print('>', 'Policy: {}'.format(args.policy))
    print('>', 'Number of enrolment templates: {}'.format(args.n_templates))
    print('>', 'Number of joint attacks: {}'.format(args.n_attacks))

    print('>', 'Test pairs dataset path: {}'.format(args.sv_base_path))
    print('>', 'Test pairs path: {}'.format(args.sv_pair_path))
    print('>', 'Number of test pairs: {}'.format(args.sv_n_pair))

    print('>', 'Test dataset path: {}'.format(args.sv_base_path))
    print('>', 'Test pairs path: {}'.format(args.sv_pair_path))
    print('>', 'Number of test pairs: {}'.format(args.sv_n_pair))

    print('>', 'Master voice base path: {}'.format(args.mv_base_path))
    print('>', 'Master voice meta path: {}'.format(args.mv_meta))
    print('>', 'Audio meta path: {}'.format(args.audio_meta))
    print('>', 'Number of samples per template: {}'.format(args.n_templates))

    print('>', 'Master voice population path: {}'.format(args.mv_set))

    print('>', 'Sample rate: {}'.format(args.sample_rate))
    print('>', 'Maximum number of seconds: {}'.format(args.n_seconds))
    print('>', 'Noise dir: {}'.format(args.noise_dir))
    print('>', 'Augmentation flag: {}'.format(args.augment))

    assert '/v' in args.net_verifier and '/v' in args.net_gan and '/v' in args.mv_set

    # Load noise data
    print('Load impulse response paths')
    noise_paths = load_noise_paths(args.noise_dir)
    print('Cache impulse response data')
    noise_cache = cache_noise_data(noise_paths, sample_rate=args.sample_rate)

    # Create and restore model
    print('Creating model')
    available_nets = {'xvector': XVector, 'vggvox': VggVox, 'resnet50vox': ResNet50Vox, 'resnet34vox': ResNet34Vox}
    model = available_nets[args.net.split('/')[0]](id=int(args.net.split('/')[1].replace('v','')), noises=noise_paths, cache=noise_cache, n_seconds=args.n_seconds, sample_rate=args.sample_rate)

    if not os.path.exists(args.sv_pair_path):
        print('Creating trials file with templates', args.n_templates)
        create_template_trials(args.sv_base_path, args.sv_pair_path, args.n_templates, args.sv_n_pair, args.sv_n_pair)
        print('> trial pairs file saved')

    # Retrieve thresholds
    print('Retrieve verification thresholds')
    test_data = load_test_data_from_file(args.sv_base_path, args.sv_pair_path, args.sv_n_pair, args.sample_rate, args.n_seconds)
    (_, _, _, thr_eer), (_, _, thr_far1) = model.test(test_data)

    # Load data for impersonation test
    x_mv_test, y_mv_test, male_x_mv_test, female_x_mv_test = load_mv_data(args.mv_meta, args.mv_base_path, args.audio_meta, args.sample_rate, args.n_seconds, args.n_templates)

    # Test model against impersonation, varying population, master voice sample, verification policy {any/avg}, verification threshold {eer/far1}, target gender {m/f}
    print('Testing master voice impersonation')
    print("{:<15} {:<23} {:<23}".format('', 'EER', 'FAR1%'))
    print("{:<15} {:<11} {:<11} {:<11} {:<11}".format('', 'ANY', 'AVG', 'ANY', 'AVG'))
    print("{:<15} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5}".format('', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F'))

    results = {}
    eer_any_m = []
    eer_any_f = []
    eer_avg_m = []
    eer_avg_f = []
    far1_any_m = []
    far1_any_f = []
    far1_avg_m = []
    far1_avg_f = []
    for mv_file_index, mv_file in enumerate(os.listdir(os.path.join('./data/vs_mv_data', args.mv_set))):

        if mv_file.endswith('.wav'):

            results[mv_file] = {}
            mv_signal = decode_audio(os.path.join('./data/vs_mv_data', args.mv_set, mv_file), tgt_sample_rate=args.sample_rate).reshape((-1, 1))[:args.sample_rate*args.n_seconds, :]
            for policy in ['any', 'avg']:
                results[mv_file][policy] = {}
                for thr_type, thr in {'eer': thr_eer, 'far1': thr_far1}.items():
                    results[mv_file][policy][thr_type] = model.impersonate(mv_signal, thr, policy, x_mv_test, y_mv_test, male_x_mv_test, female_x_mv_test, args.n_templates)

            eer_any_m.append(len([1 for fac in results[mv_file]['any']['eer']['m'] if fac > 0]) / (len(male_x_mv_test) + len(female_x_mv_test)))
            eer_any_f.append(len([1 for fac in results[mv_file]['any']['eer']['f'] if fac > 0]) / (len(male_x_mv_test) + len(female_x_mv_test)))
            eer_avg_m.append(len([1 for fac in results[mv_file]['avg']['eer']['m'] if fac > 0]) / (len(male_x_mv_test) + len(female_x_mv_test)))
            eer_avg_f.append(len([1 for fac in results[mv_file]['avg']['eer']['f'] if fac > 0]) / (len(male_x_mv_test) + len(female_x_mv_test)))
            far1_any_m.append(len([1 for fac in results[mv_file]['any']['far1']['m'] if fac > 0]) / (len(male_x_mv_test) + len(female_x_mv_test)))
            far1_any_f.append(len([1 for fac in results[mv_file]['any']['far1']['f'] if fac > 0]) / (len(male_x_mv_test) + len(female_x_mv_test)))
            far1_avg_m.append(len([1 for fac in results[mv_file]['avg']['far1']['m'] if fac > 0]) / (len(male_x_mv_test) + len(female_x_mv_test)))
            far1_avg_f.append(len([1 for fac in results[mv_file]['avg']['far1']['f'] if fac > 0]) / (len(male_x_mv_test) + len(female_x_mv_test)))
            print("{:<15}".format(mv_file), end=' ')
            print("%0.3f %0.3f %0.3f %0.3f %0.3f %0.3f %0.3f %0.3f" % (eer_any_m[-1], eer_any_f[-1], eer_avg_m[-1], eer_avg_f[-1], far1_any_m[-1], far1_any_f[-1], far1_avg_m[-1], far1_avg_f[-1]))

            with open(os.path.join('./data/vs_mv_data', args.mv_set, 'results.json'), 'w') as fp:
                json.dump(results, fp)
                print('Saved result file till', mv_file_index, end='\n')

    # Print average impersonation rates
    eer_any_m = float(np.mean(eer_any_m))
    eer_any_f = float(np.mean(eer_any_f))
    eer_avg_m = float(np.mean(eer_avg_m))
    eer_avg_f = float(np.mean(eer_avg_f))
    far1_any_m = float(np.mean(far1_any_m))
    far1_any_f = float(np.mean(far1_any_f))
    far1_avg_m = float(np.mean(far1_avg_m))
    far1_avg_f = float(np.mean(far1_avg_f))
    print("%0.3f %0.3f %0.3f %0.3f %0.3f %0.3f %0.3f %0.3f" % (eer_any_m, eer_any_f, eer_avg_m, eer_avg_f, far1_any_m, far1_any_f, far1_avg_m, far1_avg_f))


if __name__ == '__main__':
    main()