import os
import argparse

os.environ["PYTHONPATH"] = '".":"./rtvc"'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

# Parse arguments
parser = argparse.ArgumentParser(description='Run all experiments (Transferability)')
parser.add_argument('-g', dest='gender', default='female', type=str, action='store', choices=('female', 'male')) 
parser.add_argument('-p', dest='playback', action='store_true') 
parser.add_argument('-a', dest='attack', default='pgd', type=str, action='store', choices=('nes', 'pgd')) 
parser.add_argument('-t', dest='target', default='attack', type=str, action='store', choices=('attack', 'test')) 
args = parser.parse_args()

# Experiment setup
batch_size = 256
n = -10

# NES settings: nn - number of samples; ns - sigma (bandwidth of the search distribution)
nn = 100
ns = 0.001

epsilons = (0.001, 0.005, 0.01)

gradient = 'normed'
gender = args.gender
attack = f'{args.attack}@wave'
play_flags = '--play' if args.playback else ''
encoders = [f'{se}/v000' for se in ('vggvox', 'resnet50', 'resnet34', 'thinresnet', 'xvector')]

# Run attacks (PGD gradient)
if args.target == 'attack' and gradient == 'pgd':

    for se in encoders:

        if 'xvector' in se and args.attack == 'pgd':
            continue

        for e in epsilons:
            cmd = f'python3 routines/mv/optimize.py --netv {se} --dataset interspeech --seed ./data/vs_mv_seed/{gender}-10/ --attack {attack} --gender {gender} --gradient pgd --n_steps {n} --step_size {e/10} --clip_av {e} --batch {batch_size} --nes_n {nn} --nes_sigma {ns} --results_dir "results_transfer"'
            os.system(cmd)

# Run attacks (normalized gradient)
if args.target == 'attack' and gradient == 'normed':

    for se in encoders:

        if 'xvector' in se and args.attack == 'pgd':
            continue

        for e in epsilons:
            cmd = f'python3 routines/mv/optimize.py --netv {se} --dataset interspeech --seed ./data/vs_mv_seed/{gender}-10/ --attack {attack} --gender {gender} --gradient normed --n_steps {n} --step_size {e} --batch {batch_size} --nes_n {nn} --nes_sigma {ns} {play_flags} --results_dir "results_transfer"'
            os.system(cmd)

# Test
if args.target == 'test':
    data = 'data/results_transfer'
    for se in encoders:
        cmd = f'python3 routines/mv/test.py --net {se} --dataset interspeech --samples {data} --policy avg --level far1'
        os.system(cmd)
