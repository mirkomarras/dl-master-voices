#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=120:00:00
#SBATCH --mem=16000GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=VGGTrain
#SBATCH --mail-type=END
#SBATCH --mail-user=m.marras19@gmail.com
#SBATCH --output=slurm_train_vggvox_%j.out

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
python $SRCDIR/src/core/verifier/tf/train.py --audio_dir "/beegfs/mm10572/voxceleb1/dev,/beegfs/mm10572/voxceleb2/dev" --val_base_path "/beegfs/mm10572/voxceleb1/test" --net "vggvox" --augment 1