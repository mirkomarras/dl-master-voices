#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import os

from helpers.dataset import load_mv_list

import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main():

    # Parameters for verifier
    n_templates = 10
    mv_base_path = 'voxceleb2'
    mv_meta = '/beegfs/mm11333/dl-master-voices/data/ad_voxceleb12/vox2_mv_data.npz'
    audio_meta = '/beegfs/mm11333/dl-master-voices/data/ad_voxceleb12/vox12_meta_data.csv'
    mv_path = '/beegfs/mm11333/data/mastervoice1'
    csv_path = '/beegfs/mm11333/voxceleb_trainer/data/pairs'

    x_mv_test, y_mv_test, g_mv_test = load_mv_list(mv_meta, mv_base_path, audio_meta, n_templates)

    print('Sample paths', x_mv_test[:2])
    print('Sample ids', y_mv_test[:2])
    print('Sample genders', g_mv_test[:2])

    for mv_set in os.listdir(mv_path):
        for mv_file in os.listdir(os.path.join(mv_path, mv_set, 'v0')):
            if '.wav' in mv_file:
                df = pd.DataFrame(list(zip(x_mv_test, y_mv_test, g_mv_test)), columns=['path1', 'id', 'gender'])
                df['path2'] = os.path.join(mv_path.split('/')[-1], mv_set, 'v0', mv_file)
                df['label'] = 0
                df = df[['label', 'path1', 'path2', 'gender']]
                df.to_csv(os.path.join(csv_path, mv_set + '__' + mv_file.split('.')[0] + '.csv'), index=False, header=False, sep=' ')
            print(os.path.join(mv_path, mv_set, 'v0', mv_file))


if __name__ == '__main__':
    main()