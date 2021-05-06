#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 20:48:36 2020

@author: pkorus
"""
import os
import sys
import json
from matplotlib import pyplot as plt
import numpy as np

# root = ''

if len(sys.argv) == 1:
    dirname = 'data/vs_mv_data/vggvox_v000_pgd_spec_f/v025/'
else:
    dirname = sys.argv[1]

if not os.path.isdir(dirname):
    print('ERROR: Directory does not exist...')

with open(os.path.join(dirname, 'stats.json')) as f:
    data = json.load(f)

fig, axes = plt.subplots(2,1)
fig.set_size_inches((4, 7))

print(f"IMP MV FAR1 = {np.mean(data['mv_far1_results'])}")
print(f"IMP MV EER = {np.mean(data['mv_eer_results'])}")
p1, x1 = np.histogram(data['mv_far1_results'], 5)
x1 = np.convolve(x1, [0.5, 0.5], 'valid')
axes[0].plot(x1, p1, label='mv @ fpr 1%')

print(f"IMP SV FAR1 = {np.mean(data['sv_far1_results'])}")
print(f"IMP SV EER = {np.mean(data['sv_eer_results'])}")
p2, x2 = np.histogram(data['sv_far1_results'], 5)
x2 = np.convolve(x2, [0.5, 0.5], 'valid')
axes[0].plot(x2, p2, label='sv @ fpr 1%')


axes[0].hist(data['mv_far1_results'], 10)
axes[0].hist(data['sv_far1_results'], 10)
axes[0].set_xlabel('impersonation rate')
axes[0].set_xlim([0, 1])
axes[0].legend()

axes[0].set_title(dirname)

axes[1].plot([0, 1], [0, 1], ':k')
axes[1].plot(data['sv_far1_results'], data['mv_far1_results'], '.')
axes[1].set_xlim([0, 1])
axes[1].set_ylim([0, 1])
axes[1].set_xlabel('sv impersonation rate')
axes[1].set_ylabel('mv impersonation rate')

fig.savefig(os.path.join(dirname, 'stats.png'))