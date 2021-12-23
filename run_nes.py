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
args = parser.parse_args()

# Experiment setup
batch_size = 256
e = 0.01
ss = 0.01
steps = (-10)
nes_n = (25, 50, 100)
nes_sigma = (0.01, 0.005, 0.001)
epsilons = (0.0005, 0.001, 0.005, 0.01, 0.05)
attack = 'nes@wave'
gender = args.gender
se = f'{args.encoder}/v000'
gradient = args.gradient

# Grid search (PGD) on 10 seed samples only
if args.target == 'grid' and gradient == 'pgd':
    for n in steps:
        for nn in nes_n:
            for ns in nes_sigma:
                cmd = f'python3 routines/mv/optimize.py --netv {se} --dataset interspeech --seed ./data/vs_mv_seed/{gender}-10/ --attack {attack} --gender {gender} --gradient pgd --n_steps {n} --step_size {e/10} --clip_av {e} --batch {batch_size} --nes_n {nn} --nes_sigma {ns} --results_dir "results/nes_grid_pgd"'
                os.system(cmd)

# Grid search (normed) on 10 seed samples only
if args.target == 'grid' and gradient == 'normed':
    for n in steps:
        for nn in nes_n:
            for ns in nes_sigma:
                cmd = f'python3 routines/mv/optimize.py --netv {se} --dataset interspeech --seed ./data/vs_mv_seed/{gender}-10/ --attack {attack} --gender {gender} --gradient normed --n_steps {n} --step_size {ss} --batch {batch_size} --nes_n {nn} --nes_sigma {ns} --results_dir "results/nes_grid_normed"'
                os.system(cmd)

# Best NES setting (supposed)
n = -10
# NES settings: nn - number of samples; ns - sigma (bandwidth of the search distribution)
nn = 100
ns = 0.001
if args.target == 'full' and gradient == 'pgd':
    for e in epsilons:
        cmd = f'python3 routines/mv/optimize.py --netv {se} --dataset interspeech --seed ./data/vs_mv_seed/{gender}/ --attack {attack} --gender {gender} --gradient pgd --n_steps {n} --step_size {e/10} --clip_av {e} --batch {batch_size} --nes_n {nn} --nes_sigma {ns} --results_dir "results/nes_pgd"'
        os.system(cmd)

if args.target == 'full' and gradient == 'normed':
    for e in epsilons:
        cmd = f'python3 routines/mv/optimize.py --netv {se} --dataset interspeech --seed ./data/vs_mv_seed/{gender}/ --attack {attack} --gender {gender} --gradient normed --n_steps {n} --step_size {e} --batch {batch_size} --nes_n {nn} --nes_sigma {ns} --results_dir "results/nes_normed"'
        os.system(cmd)

# Test
if args.target == 'test':
    for data in ('data/results/nes_grid_pgd', 'data/results/nes_grid_normed', 'data/results/nes_pgd', 'data/results/nes_normed'):
        cmd = f'python3 routines/mv/test.py --net {se} --dataset interspeech --samples {data} --policy avg --level far1'
        os.system(cmd)
