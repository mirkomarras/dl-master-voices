#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=120:00:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=GanTest
#SBATCH --mail-type=END
#SBATCH --mail-user=m.marras19@gmail.com
#SBATCH --output=jobs/slurm_test_wavegan_%j.out

module purge
module unload cuda/8.0.44
module load cuda/9.0.176
module load cudnn/9.0v7.3.0.29
module load ffmpeg/intel/3.2.2

export PRJ_PATH="${PWD}"
export NET="wavegan"
export GENDER="neutral"
export VERSION="9"

export PYTHONPATH=$PRJ_PATH
source $PRJ_PATH/mvenv/bin/activate

python -u $PRJ_PATH/routines/gan/preview.py --net $NET --gender $GENDER --version $VERSION