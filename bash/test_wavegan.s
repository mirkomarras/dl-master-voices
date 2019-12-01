#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=120:00:00
#SBATCH --mem=16000GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=MWGanTrain
#SBATCH --mail-type=END
#SBATCH --mail-user=m.marras19@gmail.com
#SBATCH --output=jobs/slurm_test_wavegan_%j.out

module purge
module unload cuda/8.0.44
module load cuda/9.0.176
module load cudnn/9.0v7.3.0.29
module load ffmpeg/intel/3.2.2

SRCDIR=/beegfs/mm10572/dl-master-voices
ENVDIR=$SRCDIR/mvenv
RUNDIR=$SRCDIR/jobs

mkdir -p $RUNDIR

export PYTHONPATH="$(SRCDIR)"
source $ENVDIR/bin/activate

cd $RUNDIR
python $SRCDIR/src/core/gan/tf/preview.py --net "wavegan" --gender "male" --version "9"