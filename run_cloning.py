import os
import argparse

os.environ["PYTHONPATH"] = '.:./rtvc'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

# Parse arguments
parser = argparse.ArgumentParser(description='Run all experiments (Voice cloning)')
parser.add_argument('-e', dest='encoder', default='vggvox', type=str, action='store')
parser.add_argument('-g', dest='gender', default='female', type=str, action='store', choices=('female', 'male'))
parser.add_argument('-t', dest='target', default='full', type=str, action='store', choices=('grid', 'full', 'test'))
args = parser.parse_args()

# Experiment setup
batch_size = 16
n = 10

# NES settings grid:
nes_n = (20, 40)
nes_sigma = (0.025, 0.01, 0.005)

# NES settings: nn - number of samples; ns - sigma (bandwidth of the search distribution)
nn = 20
ns = 0.025

epsilons = (0.005, 0.01, 0.05)

gradient = 'normed'
attack = 'nes@cloning'
gender = args.gender
se = f'{args.encoder}/v000'
encoders = [f'{se}/v000' for se in ('vggvox', 'resnet50', 'resnet34', 'thinresnet', 'xvector')]

# Grid search (normed) on 10 seed samples only
if args.target == 'grid' and gradient == 'normed':
    for ss in epsilons:
        for nn in nes_n:
            for ns in nes_sigma:
                cmd = f'python3 routines/mv/optimize.py --netv {se} --dataset interspeech --seed ./data/vs_mv_seed/{gender}/001.wav --attack {attack} --gender {gender} --gradient normed --n_steps {n} --step_size {ss} --batch {batch_size} --nes_n {nn} --nes_sigma {ns} --results_dir "results/cloning_grid"'
                os.system(cmd)

# Run full attack
if args.target == 'full' and gradient == 'normed':
    ss = 0.01
    nn = 20
    ns = 0.025
    cmd = f'python3 routines/mv/optimize.py --netv {se} --dataset interspeech --seed ./data/vs_mv_seed/{gender}/001.wav --attack {attack} --gender {gender} --gradient normed --n_steps {n} --step_size {ss} --batch {batch_size} --nes_n {nn} --nes_sigma {ns} --results_dir "results/cloning"'
    os.system(cmd)

# Test
if args.target == 'test':
    for data in ('results/cloning', 'results/cloning_grid'):
        for se in encoders:
            cmd = f'python3 routines/mv/test.py --net {se} --dataset interspeech --samples {data} --policy avg --level far1'
            os.system(cmd)
