import os

batch_size = 256
e = 0.01
ss = 0.01
steps = (-10)
nes_n = (25, 50, 100)
nes_sigma = (0.01, 0.005, 0.001)
attack = 'nes@wave'
gender = 'female'
se = 'vggvox/v000'

# Grid search (PGD)
for n in steps:
    for nn in nes_n:
        for ns in nes_sigma:
            cmd = f'python3 routines/mv/optimize.py --netv {se} --dataset interspeech --seed ./data/vs_mv_seed/{gender}-10/ --attack {attack} --gender {gender} --gradient pgd --n_steps {n} --epsilon {e} --clip_av {e} --batch {batch_size} --nes_n {nn} --nes_sigma {ns} --results_dir "results_nes_grid_pgd"'
            os.system(cmd)

# Grid search (normed)
for n in steps:
    for nn in nes_n:
        for ns in nes_sigma:
            cmd = f'python3 routines/mv/optimize.py --netv {se} --dataset interspeech --seed ./data/vs_mv_seed/{gender}-10/ --attack {attack} --gender {gender} --gradient normed --n_steps {n} --step_size {ss} --batch {batch_size} --nes_n {nn} --nes_sigma {ns} --results_dir "results_nes_grid_normed"'
            os.system(cmd)

# Best setting (supposed)
n = -10
nn = 100
ns = 0.001
cmd = f'python3 routines/mv/optimize.py --netv {se} --dataset interspeech --seed ./data/vs_mv_seed/{gender}/ --attack {attack} --gender {gender} --gradient pgd --n_steps {n} --epsilon {e} --clip_av {e} --batch {batch_size} --nes_n {nn} --nes_sigma {ns} --results_dir "results_nes_pgd"'
os.system(cmd)

cmd = f'python3 routines/mv/optimize.py --netv {se} --dataset interspeech --seed ./data/vs_mv_seed/{gender}/ --attack {attack} --gender {gender} --gradient normed --n_steps {n} --step_size {ss} --batch {batch_size} --nes_n {nn} --nes_sigma {ns} --results_dir "results_nes_normed"'
os.system(cmd)
