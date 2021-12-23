#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 13:22:27 2021

@author: pkorus
"""



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 20:48:36 2020

@author: pkorus
"""
import os
import sys
import json
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


from helpers import results as hr

df = hr.results_df('data/results/results_plain_pgd/vggvox_v000_pgd_wave_f')

print(df.to_string())

# %%

c1 = 'step_size_override'
c2 = 'clip_av'

n1 = len(df[c1].unique())
n2 = len(df[c2].unique())

plt.imshow(df['far1-mv'].values.reshape((n2, n1)))
plt.xticks(np.arange(0,n1), df[c1].unique())
plt.yticks(np.arange(0,n2), df[c2].unique())
plt.colorbar()
plt.xlabel(c1)
plt.ylabel(c2)

# %%

subdir = 'v000'

progress = {x: y for x, y in np.load(os.path.join(dirname, subdir, 'opt_progress_001.npz'), allow_pickle=True).items()}

plt.plot([x['f'] for x in progress['mv_far1_results']])
plt.title(f'{subdir}')
plt.xlabel('Steps')
plt.ylabel('far1 impersonation rate')

# %%

from collections import OrderedDict

dirname = 'data/vs_mv_data/vggvox_v000_pgd_wave_f_L2_sgn_nosgd'

results = OrderedDict()
labels = {}

scatter = []

for x in range(7):
    subdir = f'v{x:03d}'
    progress = {x: y for x, y in np.load(os.path.join(dirname, subdir, 'opt_progress_001.npz'), allow_pickle=True).items()}
    results[subdir] = [x['f'] for x in progress['mv_far1_results']]
    # results[subdir] = progress['l2_norm']
    
    with open(os.path.join(dirname, subdir, 'params.txt')) as f:
        params = f.read()
        args = eval(eval(params))
        
    labels[subdir] = {
        'step': args.step_size_override,
        'sgd': x > 3,
        'label': f'{args.step_size_override} sgd={x>3}'
    }
    
    scatter.append((progress['max_dist'][-1], results[subdir][-1]))

# %%

for k in results:
    plt.plot(results[k], label=labels[k]['label'], linestyle='-' if labels[k]['sgd'] else ':')
    
plt.legend()
plt.xticks(np.arange(0,18))
plt.xlabel('Epoch')
plt.ylabel('far-1 IR for females')

# %%

for i, k in enumerate(results):
    plt.plot(scatter[i][0], scatter[i][1], linestyle='', marker='o' if labels[k]['sgd'] else 'x',
             label=labels[k]['label'])
    
plt.legend()
plt.xlabel('max $|v|$')
plt.ylabel('far-1 IR for females')
