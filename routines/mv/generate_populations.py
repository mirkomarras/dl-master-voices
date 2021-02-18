#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from scipy.spatial.distance import cosine
from itertools import product
import tensorflow as tf
import pandas as pd
import numpy as np
import argparse

from helpers.dataset import generate_enrolled_samples

from models import verifier

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main():
    parser = argparse.ArgumentParser(description='Create master voice training and test populations')

    parser.add_argument('--pop', dest='pop', default='train/20/20,test/20/10', type=str, action='store', help='Comma-separated list of populations in the form popname/nusers/nuttrs')

    settings = vars(parser.parse_args())

    prepr = {}
    for pop_info in settings['pop'].split(','):
        pop_instance = pop_info.split('/')
        prepr[pop_instance[0]] = {'nusers': int(pop_instance[1]), 'nuttrs': int(pop_instance[2])}

    generate_enrolled_samples(prepr, 'debug')

if __name__ == '__main__':
    main()

