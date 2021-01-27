#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from models import verifier
from models import mv
from helpers import dataset
from helpers import datapipeline

from collections import namedtuple


def test_siamese_model():

    GENDER = 'female'
    SV_MODEL = 'vggvox/v004'

    AUDIO_DIR = 'data/voxceleb2/dev'
    AUDIO_META = 'data/vs_mv_pairs/meta_data_vox12_all.csv'
    MV_POPULATION_FILE = 'data/vs_mv_pairs/data_mv_vox2_debug.npz'

    SEED_DIR = 'tests/sv'
    OUT_DIR = 'tests/sv_out'  # data/vs_mv_data/vggvox-v004_real_u-f

    USER_IDS = dataset.get_mv_analysis_users(MV_POPULATION_FILE, type='train')

    Params = namedtuple('SiameseParams', 'mv_gender')
    params = Params(mv_gender=GENDER)

    siamese_model = mv.SiameseModel(dir=OUT_DIR, params=params, playback=False, ir_dir=None)
    siamese_model.set_verifier(verifier.get_model(SV_MODEL))
    siamese_model.build()

    x_train, y_train = dataset.load_data_set([AUDIO_DIR], USER_IDS, include=True, n_samples=32)
    x_train, y_train = dataset.filter_by_gender(x_train, y_train, AUDIO_META, GENDER)

    assert len(x_train) == 160
    assert len(y_train) == 160

    train_data = datapipeline.data_pipeline_mv(x_train, y_train, int(16000*2.57), 16000, 32, 128, 'spectrum')
    test_data = dataset.load_mv_data(MV_POPULATION_FILE, AUDIO_DIR, AUDIO_META, 16000, 10, 'test')

    stats = siamese_model.batch_optimize_by_path(SEED_DIR, train_data, test_data, [0.9, 0.8])

    assert 'mv_eer_results' in stats, "MV optimization results are incomplete"
    assert len(stats['mv_eer_results']) == 2, "Expecting results for 2 master voices"
    assert stats['max_dist'][0] > 0, "Adversarial distortion should not be zero"
