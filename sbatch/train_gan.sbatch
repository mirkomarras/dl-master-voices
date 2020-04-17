#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=168:00:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=MalGan
#SBATCH --mail-type=END
#SBATCH --mail-user=m.marras19@gmail.com
#SBATCH --output=jobs/slurm_train_mal_gan_%j.out

module purge
module unload cuda/8.0.44
module load cuda/10.0.130
module load cudnn/10.0v7.6.2.24
module load ffmpeg/intel/3.2.2

export PRJ_PATH="${PWD}"
export AUDIO_DIR="/beegfs/mm11333/data/voxceleb1/dev,/beegfs/mm11333/data/voxceleb2/dev"
export NET="wavegan"
export GENDER="female"
export SLICE_LEN=65536

export PYTHONPATH=$PRJ_PATH
source $PRJ_PATH/mvenv/bin/activate

python -u $PRJ_PATH/routines/gan/train.py --net $NET --audio_dir $AUDIO_DIR --gender $GENDER --slice_len $SLICE_LEN

