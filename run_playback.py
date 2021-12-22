import os

batch_size = 256
epsilons = (0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1)
steps = (10,)
attack = 'pgd@wave'
gradient = 'pgd'
gender = 'female'
se = 'vggvox/v000'

# L_inf version
for n in steps:
    for e in epsilons:
        s = e / 10
        cmd = f'python3 routines/mv/optimize.py --netv {se} --dataset interspeech --seed ./data/vs_mv_seed/{gender}/001.wav --attack {attack} --gender {gender} --gradient {gradient} --n_steps {n} --step_size {s} --clip_av {e} --batch {batch_size} --play --results_dir "results_play_pgd"'
        os.system(cmd)

# L_2 version
gradient = 'normed'
step_sizes = (0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1)

for n in steps:
    for s in step_sizes:
        cmd = f'python3 routines/mv/optimize.py --netv {se} --dataset interspeech --seed ./data/vs_mv_seed/{gender}/001.wav --attack {attack} --gender {gender} --gradient {gradient} --n_steps {n} --step_size {s} --batch {batch_size} --play --results_dir "results_play_normed"'
        os.system(cmd)

# Tests
for model in ('vggvox/v000', 'resnet50/v000', 'xvector/v000'):
    for data in ('data/results_play_normed/vggvox_v000_pgd_wave_f', 'data/results_play_pgd/vggvox_v000_pgd_wave_f'):
        cmd = f'python3 routines/mv/test.py --net {model} --dataset interspeech --samples {data} --policy avg --level far1'
        os.system(cmd)
