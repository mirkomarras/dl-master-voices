import os
import argparse

os.environ["PYTHONPATH"] = '.:./rtvc'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

# Parse arguments
parser = argparse.ArgumentParser(description='Run all experiments (NES)')
parser.add_argument('-e', dest='encoder', default='vggvox', type=str, action='store') 
parser.add_argument('-g', dest='gender', default='female', type=str, action='store', choices=('female', 'male')) 
parser.add_argument('-d', dest='gradient', default='normed', type=str, action='store', choices=('normed', 'pgd')) 
parser.add_argument('-t', dest='target', default='grid', type=str, action='store', choices=('grid', 'full', 'test')) 
parser.add_argument('-s', dest='step', default='0.01', type=float, action='store') 
parser.add_argument('-p', dest='playback', action='store_true') 
args = parser.parse_args()

# Experiment setup
batch_size = 256
e = args.step
ss = 0.01
steps = (-10,)
nes_n = (25, 50, 100)
nes_sigma = (0.01, 0.005, 0.001)
epsilons = (e, )
attack = 'nes@wave'
gender = args.gender
se = f'{args.encoder}/v000'
gradient = args.gradient
extra_params = ''

play_flags = '--play' if args.playback else ''
play_suffix = '_play' if args.playback else ''

# Grid
# 3 sigma x 10 steps x (10 + 5 + 3 min) = 9h / seed  -> 27h for 3 eps

# Full
# 670 s / epoch at NES_n=100 = 10 min
# 1 eps x 100 seeds * 10 steps * 10 min ~= 7 days / eps

# Grid search (PGD) on 10 seed samples only
if args.target == 'grid' and gradient == 'pgd':
    for n in steps:
        for nn in nes_n:
            for ns in nes_sigma:
                cmd = f'python3 routines/mv/optimize.py --netv {se} --dataset interspeech --seed ./data/vs_mv_seed/{gender}-10/ --attack {attack} --gender {gender} --gradient pgd --n_steps {n} --step_size {e/10} --clip_av {e} --batch {batch_size} --nes_n {nn} --nes_sigma {ns} --results_dir "results/nes_grid_pgd{play_suffix}" {play_flags}'
                os.system(cmd)

# Grid search (normed) on 10 seed samples only
if args.target == 'grid' and gradient == 'normed':
    for n in steps:
        for nn in nes_n:
            for ns in nes_sigma:
                cmd = f'python3 routines/mv/optimize.py --netv {se} --dataset interspeech --seed ./data/vs_mv_seed/{gender}-10/ --attack {attack} --gender {gender} --gradient normed --n_steps {n} --step_size {ss} --batch {batch_size} --nes_n {nn} --nes_sigma {ns} --results_dir "results/nes_grid_normed{play_suffix}" {play_flags}'
                os.system(cmd)

# Best NES setting (supposed)
n = -10
# NES settings: nn - number of samples; ns - sigma (bandwidth of the search distribution)
nn = 100
ns = 0.001
if args.target == 'full' and gradient == 'pgd':
    for e in epsilons:
        cmd = f'python3 routines/mv/optimize.py --netv {se} --dataset interspeech --seed ./data/vs_mv_seed/{gender}/ --attack {attack} --gender {gender} --gradient pgd --n_steps {n} --step_size {e/10} --clip_av {e} --batch {batch_size} --nes_n {nn} --nes_sigma {ns} --results_dir "results/nes_pgd{play_suffix}" {extra_params} {play_flags}'
        os.system(cmd)

if args.target == 'full' and gradient == 'normed':
    for e in epsilons:
        cmd = f'python3 routines/mv/optimize.py --netv {se} --dataset interspeech --seed ./data/vs_mv_seed/{gender}/ --attack {attack} --gender {gender} --gradient normed --n_steps {n} --step_size {e} --batch {batch_size} --nes_n {nn} --nes_sigma {ns} --results_dir "results/nes_normed{play_suffix}" {extra_params} {play_flags}'
        os.system(cmd)

# Test
if args.target == 'test':
    for data in ('data/results/nes_grid_pgd', 'data/results/nes_grid_normed', 'data/results/nes_pgd', 'data/results/nes_normed', 'data/results/nes_grid_pgd_play', 'data/results/nes_grid_normed_play', 'data/results/nes_pgd_play', 'data/results/nes_normed_play'):
    # for data in ('data/results/nes_normed_play', ):
        for se in ('vggvox/v000', 'thin_resnet/v000', 'resnet50/v000', 'xvector/v000'):
            cmd = f'python3 routines/mv/test.py --net {se} --dataset interspeech --samples {data} --policy avg --level far1'
            os.system(cmd)
