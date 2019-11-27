#!/bin/bash

# Request resource
srun --time=120:00:00 --ntasks-per-node=1 --mem=64000 --gres=gpu:1 --pty /bin/bash

# Load modules
module unload cuda/8.0.44
module load cuda/9.0.176
module load cudnn/9.0v7.3.0.29
module load ffmpeg/intel/3.2.2

# Activate env
source ../mvenv/bin/activate

# Train
python ./src/core/verifier/tf/train.py --audio_dir "/beegfs/mm10572/voxceleb1/dev" --val_base_path "/beegfs/mm10572/voxceleb1/test" --net "vggvox" --augment 1

