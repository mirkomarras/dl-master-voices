import os

batch_size = 16
epsilons = (0.001, 0.01, 0.1)
steps = (1, 3, 5, 10)
attack = 'pgd@wave'
gender = 'female'
se = 'vggvox/v000'

for e in epsilons:
    for n in steps:
        cmd = f'python3 routines/mv/optimize.py --netv {se} --dataset interspeech --seed ./data/vs_mv_seed/{gender}/001.wav --attack {attack} --gender {gender} --gradient pgd --n_steps {n} --epsilon {e} --clip_av {e} --batch {batch_size} --memory-growth'
        os.system(cmd)

