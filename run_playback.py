import os

batch_size = 256
epsilons = (0.001, 0.01, 0.1)
steps = (10,)
attack = 'pgd@wave'
gradient = 'pgd'
gender = 'female'
se = 'vggvox/v000'

# L_inf version
for n in steps:
    for e in epsilons:
        s = e / 10
        cmd = f'python3 routines/mv/optimize.py --netv {se} --dataset interspeech --seed ./data/vs_mv_seed/{gender}/001.wav --attack {attack} --gender {gender} --gradient {gradient} --n_steps {n} --step_size {s} --clip_av {e} --batch {batch_size} --play'
        os.system(cmd)

# L_2 version
gradient = 'normed'
step_sizes = (0.001, 0.01, 0.1, 1)

for n in steps:
    for s in step_sizes:
        cmd = f'python3 routines/mv/optimize.py --netv {se} --dataset interspeech --seed ./data/vs_mv_seed/{gender}/001.wav --attack {attack} --gender {gender} --gradient {gradient} --n_steps {n} --step_size {s} --batch {batch_size} --play'
        os.system(cmd)

