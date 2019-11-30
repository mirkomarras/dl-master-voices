#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=120:00:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=FWGanTrain
#SBATCH --mail-type=END
#SBATCH --mail-user=m.marras19@gmail.com
#SBATCH --output=jobs/slurm_train_wavegan_female_%j.out

module purge
module unload cuda/8.0.44
module load cuda/9.0.176
module load cudnn/9.0v7.3.0.29
module load ffmpeg/intel/3.2.2

export PYTHONPATH="/beegfs/mm10572/dl-master-voices"
source /beegfs/mm10572/dl-master-voices/mvenv/bin/activate

python /beegfs/mm10572/dl-master-voices/src/core/gan/tf/train.py --net "wavegan" --audio_dir "/beegfs/mm10572/voxceleb1/dev/beegfs/mm10572/voxceleb2/dev" --gender "female"