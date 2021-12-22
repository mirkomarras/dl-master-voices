import os
import argparse

os.environ["PYTHONPATH"] = '".":"./rtvc"'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

# Parse arguments
parser = argparse.ArgumentParser(description='Run all experiments (Playback)')
parser.add_argument('-e', dest='encoder', default='vggvox', type=str, action='store') 
parser.add_argument('-g', dest='gender', default='female', type=str, action='store', choices=('female', 'male')) 
parser.add_argument('-d', dest='gradient', default='normed', type=str, action='store', choices=('normed', 'pgd')) 
parser.add_argument('-t', dest='test_only', default=False, action='store_true') 
args = parser.parse_args()

# Experiment setup
batch_size = 256
epsilons = (0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1)
steps = (10,)
attack = 'pgd@wave'
gradient = args.gradient
gender = args.gender
se = f'{args.encoder}/v000'

# L_inf version
# 7 x 100 seeds x 10 x 10 / 60 / 60 = 19h
if gradient == 'pgd' and not args.test_only:
    for n in steps:
        for e in epsilons:
            s = e / 10
            cmd = f'python3 routines/mv/optimize.py --netv {se} --dataset interspeech --seed ./data/vs_mv_seed/{gender}-10/ --attack {attack} --gender {gender} --gradient {gradient} --n_steps {n} --step_size {s} --clip_av {e} --batch {batch_size} --play --results_dir "results_play_pgd"'
            os.system(cmd)

# L_2 version
if gradient == 'normed' and not args.test_only:
    for n in steps:
        for s in epsilons:
            cmd = f'python3 routines/mv/optimize.py --netv {se} --dataset interspeech --seed ./data/vs_mv_seed/{gender}-10/ --attack {attack} --gender {gender} --gradient {gradient} --n_steps {n} --step_size {s} --batch {batch_size} --play --results_dir "results_play_normed"'
            os.system(cmd)

# Tests
if args.test_only:
    for model in ('vggvox/v000', 'resnet50/v000', 'xvector/v000'):
        for data in ('data/results_play_normed', 'data/results_play_pgd/'):
            cmd = f'python3 routines/mv/test.py --net {model} --dataset interspeech --samples {data} --policy avg --level far1'
            os.system(cmd)

