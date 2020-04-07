#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import time
import os

from helpers.dataset import load_test_data_from_file, create_template_trials

from models.verifier.vggvox import VggVox
from models.verifier.xvector import XVector
from models.verifier.resnet50vox import ResNet50Vox
from models.verifier.resnet34vox import ResNet34Vox

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main():

    parser = argparse.ArgumentParser(description='Tensorflow speaker verification model training')

    # Parameters for verifier
    parser.add_argument('--net', dest='net', default='', type=str, action='store', help='Network model architecture')
    parser.add_argument('--policy', dest='policy', default='any', type=str, action='store', help='Verification policy')
    parser.add_argument('--n_templates', dest='n_templates', default=1, type=int, action='store', help='Number of enrolment templates')

    # Paremeters for verifier internal mechanics
    parser.add_argument('--classes', dest='classes', default=5205, type=int, action='store', help='Classes')
    parser.add_argument('--loss', dest='loss', default='softmax', type=str, choices=['softmax', 'amsoftmax'], action='store', help='Type of loss')
    parser.add_argument('--aggregation', dest='aggregation', default='avg', type=str, choices=['avg', 'vlad', 'gvlad'], action='store', help='Type of aggregation')
    parser.add_argument('--vlad_clusters', dest='vlad_clusters', default=12, type=int, action='store', help='Number of vlad clusters')
    parser.add_argument('--ghost_clusters', dest='ghost_clusters', default=2, type=int, action='store', help='Number of ghost clusters')
    parser.add_argument('--weight_decay', dest='weight_decay', default=1e-4, type=float, action='store', help='Weight decay')

    # Parameters for testing a verifier against eer
    parser.add_argument('--test_base_path', dest='test_base_path', default='./data/vs_voxceleb1/test', type=str, action='store', help='Base path for validation trials')
    parser.add_argument('--test_pair_path', dest='test_pair_path', default='./data/ad_voxceleb12/vox1_trial_pairs.csv', type=str, action='store', help='CSV file label, path_1, path_2 triplets')
    parser.add_argument('--test_n_pair', dest='test_n_pair', default=0, type=int, action='store', help='Number of test pairs')

    # Parameters for raw audio
    parser.add_argument('--sample_rate', dest='sample_rate', default=16000, type=int, action='store', help='Sample rate audio')
    parser.add_argument('--n_seconds', dest='n_seconds', default=3, type=int, action='store', help='Segment lenght in seconds')

    args = parser.parse_args()

    print('Parameters summary')

    mode = ('filterbank' if args.net.split('/')[0] == 'xvector' else 'spectrum')

    print('>', 'Net: {}'.format(args.net))
    print('>', 'Policy: {}'.format(args.policy))
    print('>', 'Number of enrolment templates: {}'.format(args.n_templates))
    print('>', 'Mode: {}'.format(mode))
    print('>', 'Classes: {}'.format(args.classes))
    print('>', 'Loss: {}'.format(args.loss))
    print('>', 'Aggregation: {}'.format(args.aggregation))

    print('>', 'Sample rate: {}'.format(args.sample_rate))
    print('>', 'Test pairs dataset path: {}'.format(args.test_base_path))
    print('>', 'Test pairs path: {}'.format(args.test_pair_path))
    print('>', 'Number of test pairs: {}'.format(args.test_n_pair))
    print('>', 'Max number of seconds: {}'.format(args.n_seconds))

    assert '/v' in args.net

    if not os.path.exists(args.test_pair_path):
        print('Creating trials file with templates', args.n_templates)
        create_template_trials(args.test_base_path, args.test_pair_path, args.n_templates, args.test_n_pair, args.test_n_pair)
        print('> trial pairs file saved')

    # Load test data
    test_data = load_test_data_from_file(args.test_base_path, args.test_pair_path, args.test_n_pair, args.n_templates, args.sample_rate, args.n_seconds)

    # Create model
    print('Creating model')
    available_nets = {'xvector': XVector, 'vggvox': VggVox, 'resnet50vox': ResNet50Vox, 'resnet34vox': ResNet34Vox}
    model = available_nets[args.net.split('/')[0]](id=int(args.net.split('/')[1].replace('v','')), n_seconds=args.n_seconds, sample_rate=args.sample_rate)
    model.build(classes=args.classes, loss=args.loss, aggregation=args.aggregation, vlad_clusters=args.vlad_clusters, ghost_clusters=args.ghost_clusters, weight_decay=args.weight_decay)
    model.load()

    # Test model
    print('Testing model')
    t1 = time.time()
    results = model.test(test_data, mode, args.policy)
    t2 = time.time()
    print('>', t2-t1, 'seconds for testing with', results)

if __name__ == '__main__':
    main()