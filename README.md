# Master Voice Toolbox
[![Build Status](https://travis-ci.org/pages-themes/cayman.svg?branch=master)](https://travis-ci.org/pages-themes/cayman)
[![GitHub version](https://badge.fury.io/gh/boennemann%2Fbadges.svg)](http://badge.fury.io/gh/boennemann%2Fbadges)
[![Dependency Status](https://david-dm.org/boennemann/badges.svg)](https://david-dm.org/boennemann/badges)
[![Open Source Love](https://badges.frapsoft.com/os/gpl/gpl.svg?v=102)](https://github.com/ellerbrock/open-source-badge/)

[Mirko Marras](https://www.mirkomarras.com/)<sup>1</sup>, [Paweł Korus](http://kt.agh.edu.pl/~korus/)<sup>2,3</sup>, 
[Nasir Memon](https://engineering.nyu.edu/faculty/nasir-memon)<sup>2</sup>, [Gianni Fenu](http://people.unica.it/giannifenu/)<sup>1</sup>
<br/><sup>1</sup> University of Cagliari, <sup>2</sup> New York University, <sup>3</sup> AGH University of Science and Technology

A Python toolbox for creating and testing impersonation capabilities of **Master Voices** (MVs), a family of adversarial 
audio samples which match large populations of speakers by chance with high probability. 

## Installation

Clone this repository:
``` 
git clone https://github.com/mirkomarras/dl-master-voices.git
cd ./dl-master-voices/
``` 

Create a Python environment:
``` 
module load python3/intel/3.6.3
python3 -m virtualenv mvenv
source mvenv/bin/activate
pip install -r requirements.txt
``` 

Copy the data folder:
``` 
unzip /archive/m/mm11333/data_20200706.zip
``` 

Create a folder for your sbatch jobs:
``` 
mkdir jobs
``` 

Add symlinks to voxceleb datasets:
``` 
ln -s /beegfs/mm11333/data/voxceleb1 ./data/
ln -s /beegfs/mm11333/data/voxceleb2 ./data/
``` 

## Getting Started

### Running scripts in interactive mode

``` 
srun --time=168:00:00 --ntasks-per-node=1 --gres=gpu:1 --mem=8000 --pty /bin/bash
export PRJ_PATH="${PWD}"
export PYTHONPATH=$PRJ_PATH
source mvenv/bin/activate
module load ffmpeg/intel/3.2.2
module load cuda/10.0.130
module load cudnn/10.0v7.6.2.24

python type/your/script/here param1 param2
``` 

### Running scripts in batch mode

Start the batch procedure:
``` 
sbatch ./sbatch/train_verifier.sbatch
``` 

Find the ```JOB ID```  of the sbatch procedure:
``` 
squeue -u $USER
``` 

Open the output file of the sbatch procedure:
``` 
cat ./jobs/slurm-<JOB ID>.out
``` 

### Running Jupyter notebooks

Run the notebook on HPC (please replace ```mm11333``` with your ```NYU ID``` at line 58 in ```run_jupyterlab_cpu.sbatch```):
``` 
cd ./notebooks/
sbatch ./run_jupyterlab_cpu.sbatch
``` 

Find the ```JOB ID``` of the notebook procedure:
``` 
squeue -u $USER
``` 

Open the output file of the sbatch procedure:
``` 
cat ./slurm-<JOB ID>.out
``` 

Find lines similar to the following ones and get the ```PORT``` (here 7500) and the Jupyter Notebook ```URL```:
``` 
 To access the notebook, open this file in a browser:
        file:///home/mm11333/.local/share/jupyter/runtime/nbserver-35214-open.html
    Or copy and paste one of these URLs:
        http://localhost:7500/?token=8d70f37561638d78b1ad0096de2ffa4abab4862d336084ae
     or http://127.0.0.1:7500/?token=8d70f37561638d78b1ad0096de2ffa4abab4862d336084ae
```
 
Open a terminal locally in your laptop and run:
``` 
ssh -L <PORT>:localhost:<PORT> <NYU ID>prince.hpc.nyu.edu
``` 

Open your browser locally and paste the ```URL``` retrieved above, here:
``` 
http://localhost:7500/?token=8d70f37561638d78b1ad0096de2ffa4abab4862d336084ae
``` 

## Using

### Train a speaker verifier
``` 
python -u ./routines/verifier/train.py --net "xvector" --learning_rate 0.001 --batch 32
```

### Test a speaker verifier
``` 
python -u ./routines/verifier/test.py --net "xvector/v000" --test_n_pair 1000
```

### Train a GAN
``` 
...
``` 

### Test a GAN
``` 
...
``` 

### Optimize a MV
``` 
...
```

### Test a MV
``` 
...
```

## Contribution
This code is provided for educational purposes and aims to facilitate reproduction of our results, and further research 
in this direction. We have done our best to document, refactor, and test the code before publication.

If you find any bugs or would like to contribute new models, training protocols, etc, please let us know.

Please feel free to file issues and pull requests on the repo and we will address them as we can.

## Citations
If you find this code useful in your work, please cite our papers:

```
Marras, M., Korus, P., Memon, N., & Fenu, G. (2019)
Adversarial Optimization for Dictionary Attacks on Speaker Verification
In: 20th Annual Conference of the International Speech Communication Association (INTERSPEECH 2019)
```

## License
This code is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This software is distributed in the hope that it will be useful, but without any warranty; without even the implied warranty of merchantability or fitness for a particular purpose. See the GNU General Public License for details.

You should have received a copy of the GNU General Public License along with this source code. If not, go the following link: http://www.gnu.org/licenses/.


