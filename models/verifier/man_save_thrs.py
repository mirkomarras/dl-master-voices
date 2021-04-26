#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from loguru import logger

import json
import os

filepath = os.path.join('./data/vs_mv_models/vggvox/v004', 'thresholds.json')
thrs = {'eer': 0.7683, 'far1': 0.8343}

with open(filepath, 'w') as thresholds_file:
    logger.info('>', 'thresholds saved in {}'.format(filepath))
    json.dump(thrs, thresholds_file)