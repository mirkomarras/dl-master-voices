import os

batch_size = 256
e = 0.01
steps = (10, 5, 1)
nes_n = (25, 50, 100, 250)
nes_sigma = (0.01, 0.005, 0.001)
attack = 'nes@wave'
gender = 'female'
se = 'vggvox/v000'

for n in steps:
    for nn in nes_n:
        for ns in nes_sigma:
            cmd = f'python3 routines/mv/optimize.py --netv {se} --dataset interspeech --seed ./data/vs_mv_seed/{gender}/001.wav --attack {attack} --gender {gender} --gradient pgd --n_steps {n} --epsilon {e} --clip_av {e} --batch {batch_size} --nes_n {nn} --nes_sigma {ns}'
            os.system(cmd)
