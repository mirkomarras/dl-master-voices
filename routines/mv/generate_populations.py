#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from scipy.spatial.distance import cosine
from itertools import product
import pandas as pd
import numpy as np
import argparse

from helpers.dataset import generate_enrolled_samples

from models import verifier

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main():
    parser = argparse.ArgumentParser(description='Split voxceleb dataset into distinct populations')

    parser.add_argument('-p', '--population', dest='population_tag', default='debug', type=str, action='store', help='Population tag, e.g. "debug"')
    parser.add_argument('-s', '--split', dest='split', default='train/20/20,test/20/10', type=str, action='store', help='Comma-separated list of populations in the form popname/nusers/nuttrs')
    parser.add_argument('-d', '--dirname', dest='dirname', default='data/voxceleb2/dev/', type=str, action='store', help='Path to the voxceleb dataset')
    parser.add_argument('-m', '--meta', dest='meta', default='data/vs_mv_pairs/meta_data_vox12_all.csv', type=str, action='store', help='Path to meta data file')

    settings = parser.parse_args()

    population_definitions = {}
    for pop_info in settings.split.split(','):
        pop_instance = pop_info.split('/')
        population_definitions[pop_instance[0]] = {'nusers': int(pop_instance[1]), 'nuttrs': int(pop_instance[2])}

    generate_enrolled_samples(population_definitions, settings.population_tag, audio_meta=settings.meta,
    audio_folder=settings.dirname)

if __name__ == '__main__':
    main()

