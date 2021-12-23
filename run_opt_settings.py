import os
import argparse

os.environ["PYTHONPATH"] = '".":"./rtvc"'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

# Parse arguments
parser = argparse.ArgumentParser(description='Run all experiments (optimization settings)')
parser.add_argument('-e', dest='encoder', default='vggvox', type=str, action='store')
parser.add_argument('-m', dest='domain', default='wave', type=str, action='store', choices=('wave', 'spec')) 
parser.add_argument('-g', dest='gender', default='female', type=str, action='store', choices=('female', 'male')) 
parser.add_argument('-d', dest='gradient', default='normed', type=str, action='store', choices=('normed', 'pgd')) 
parser.add_argument('-t', dest='test_only', default=False, action='store_true') 
args = parser.parse_args()

# Experiment setup
batch_size = 256
if args.domain == 'wave':
    epsilons = (0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05)
else:
    epsilons = (0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5)

steps = (10, 1, -10)
attack = f'pgd@{args.domain}'
gender = args.gender
se = f'{args.encoder}/v000'
gradient = args.gradient

# L_inf
# 7 x 100 seeds x 10 * 10 / 60 / 60 = 19h
# 9 x 100 seeds x (21) * 10 seconds / 60 / 60 = 52h
if gradient == 'pgd' and not args.test_only:
    for n in steps:
        for e in epsilons:
            cmd = f'python3 routines/mv/optimize.py --netv {se} --dataset interspeech --seed ./data/vs_mv_seed/{gender}/ --attack {attack} --gender {gender} --gradient {gradient} --n_steps {n} --step_size {e/10} --clip_av {e} --batch {batch_size} --results_dir "results_plain_pgd"'
            os.system(cmd)

# L_2
if gradient == 'normed' and not args.test_only:
    for n in steps:
        for e in epsilons:
            cmd = f'python3 routines/mv/optimize.py --netv {se} --dataset interspeech --seed ./data/vs_mv_seed/{gender}/ --attack {attack} --gender {gender} --gradient {gradient} --n_steps {n} --step_size {e} --batch {batch_size} --results_dir "results_plain_normed"'
            os.system(cmd)

# Testing
if args.test_only:
    for data in ('data/results_plain_pgd', 'data/results_plain_normed'):
        cmd = f'python3 routines/mv/test.py --net {se} --dataset interspeech --samples {data} --policy avg,any --level far1,eer'
        os.system(cmd)
