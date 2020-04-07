#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=168:00:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=MoVoTrain
#SBATCH --mail-type=END
#SBATCH --mail-user=m.marras19@gmail.com
#SBATCH --output=jobs/slurm_train_mv_%j.out

module purge
module unload cuda/8.0.44
module load cuda/10.0.130
module load cudnn/10.0v7.6.2.24
module load ffmpeg/intel/3.2.2

export PRJ_PATH="${PWD}"
export NET_VERIFIER="xvector/v058"
export NET_GAN="wavegan/v003"
export GENDER_GAN="female"
export GENDER_TRAIN="female"
export AUDIO_DIR="/beegfs/mm11333/data/voxceleb2/dev"
export MV_BASE_PATH="/beegfs/mm11333/data/voxceleb2"
export SV_BASE_PATH="/beegfs/mm11333/data/voxceleb1/test"

export PYTHONPATH=$PRJ_PATH
source $PRJ_PATH/mvenv/bin/activate

python -u $PRJ_PATH/routines/mv/train.py --net_verifier $NET_VERIFIER --net_gan $NET_GAN --gender_gan $GENDER_GAN --gender_train $GENDER_TRAIN --audio_dir $AUDIO_DIR --mv_base_path $MV_BASE_PATH --sv_base_path $SV_BASE_PATH
