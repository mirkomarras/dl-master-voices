#!/bin/bash

# Load modules
echo "Loading modules"
module unload cuda/8.0.44
module load cuda/9.0.176
module load cudnn/9.0v7.3.0.29
module load ffmpeg/intel/3.2.2

# Activate env
echo "Loading venv"
source ../mvenv/bin/activate

# Train
echo "Start training"
export PYTHONPATH="$(pwd)"

python ./src/core/verifier/tf/train.py --audio_dir "/beegfs/mm10572/voxceleb1/dev,/beegfs/mm10572/voxceleb2/dev" --val_base_path "/beegfs/mm10572/voxceleb1/test" --net "resnet50" --augment 1 --prefetch 100 --buffer_size 7500
