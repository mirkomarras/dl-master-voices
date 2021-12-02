import os
import shutil
from tqdm import tqdm
from pathlib import Path

# LibriSpeech
in_path = '/home/pkorus/Datasets/LibriSpeech/train-clean-100'
ou_path = '/home/pkorus/Datasets/LibriSpeech/train-clean-wav'
pattern = '.flac'
n = 20

# VCTK
in_path = '/home/pkorus/Datasets/vctk/vctk/wave/'
ou_path = '/home/pkorus/Datasets/vctk/vctk/wav-10/'
pattern = '_mic1.flac'
n = 20

selected_users = os.listdir(in_path)

print(selected_users)

for u in tqdm(selected_users):
    os.makedirs(str(Path(ou_path, u)), exist_ok=True)
    files = [str(f) for f in Path(in_path, u).glob(f'**/*{pattern}')][:n]
    for f in files:
        # of = f.replace('.flac', '.wav')
        of = os.path.join(ou_path, u, os.path.split(f)[-1].replace(pattern, '.wav'))
        command = 'ffmpeg -i {} {}'.format(f, of)
        print(f'> {command}')
        os.system(command)
