import os

batch_size = 256
epsilons = (0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05)
steps = (10, 1, -10)
attack = 'pgd@wave'
gradient = 'pgd'
gender = 'female'
se = 'vggvox/v000'

# Timings on rtx8000
# 10 s / epoch (plain)
# 11 s / epoch (playback)

# L_inf
# 7 x 100 seeds x 10 * 10 / 60 / 60 = 19h
# 9 x 100 seeds x (21) * 10 seconds / 60 / 60 = 52h
for n in steps:
    for e in epsilons:
        cmd = f'python3 routines/mv/optimize.py --netv {se} --dataset interspeech --seed ./data/vs_mv_seed/{gender}-10/ --attack {attack} --gender {gender} --gradient {gradient} --n_steps {n} --step_size {e/10} --clip_av {e} --batch {batch_size} --results_dir "results_plain_pgd"'
        os.system(cmd)

# L_2
gradient = 'normed'
for n in steps:
    for e in epsilons:
        cmd = f'python3 routines/mv/optimize.py --netv {se} --dataset interspeech --seed ./data/vs_mv_seed/{gender}-10/ --attack {attack} --gender {gender} --gradient {gradient} --n_steps {n} --step_size {e} --batch {batch_size} --results_dir "results_plain_normed"'
        os.system(cmd)

