import os

batch_size = 256
epsilons = (0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1)
steps = (1, 10, -10)
attack = 'pgd@wave'
gradient = 'pgd'
gender = 'female'
se = 'vggvox/v000'

# L_inf
for e in epsilons:
    for n in steps:
        cmd = f'python3 routines/mv/optimize.py --netv {se} --dataset interspeech --seed ./data/vs_mv_seed/{gender}/001.wav --attack {attack} --gender {gender} --gradient {gradient} --n_steps {n} --step_size {e/10} --clip_av {e} --batch {batch_size} --results_dir "results_plain_pgd"'
        os.system(cmd)

# L_2
gradient = 'normed'
for e in epsilons:
    for n in steps:
        cmd = f'python3 routines/mv/optimize.py --netv {se} --dataset interspeech --seed ./data/vs_mv_seed/{gender}/001.wav --attack {attack} --gender {gender} --gradient {gradient} --n_steps {n} --step_size {e} --batch {batch_size} --results_dir "results_plain_normed"'
        os.system(cmd)

