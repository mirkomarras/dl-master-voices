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

from argparse import Namespace


df = pd.DataFrame(columns=('epsilon', 'steps', 'step_size_override', 'clip_av', 'far1-sv', 'far1-mv', 'nes_n', 'nes_sigma'))

if len(sys.argv) == 1:
    # dirname = 'data/results/vggvox_v000_pgd_wave_f/'
    dirname = 'data/results/vggvox_v000_nes_wave_f'
else:
    dirname = sys.argv[1]

if not os.path.isdir(dirname):
    print('ERROR: Directory does not exist...')

for subdir in sorted(os.listdir(dirname))[:16]:

    with open(os.path.join(dirname, subdir, 'params.txt')) as f:
        params = f.read()
        args = eval(eval(params))
    
    with open(os.path.join(dirname, subdir, 'stats.json')) as f:
        data = json.load(f)
    
    df = df.append({
        'epsilon' : args.epsilon,
        'steps': args.n_steps,
        'step_size_override' : args.step_size_override,
        'nes_n' : args.nes_n,
        'nes_sigma' : args.nes_sigma,
        'clip_av' : args.clip_av,
        'far1-sv': np.mean(data['sv_far1_results']),
        'far1-mv': np.mean(data['mv_far1_results'])
        }, ignore_index=True)

# Drop columns with all NaNs
df = df.dropna(how='all', axis=1)

print(df)

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
