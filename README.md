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
Clone the repository:
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
unzip /archive/m/mm11333/data_original.zip
``` 

Create a folder for your sbatch jobs:
``` 
mkdir jobs
``` 

## Getting Started

### Running scripts in interactive mode

``` 
srun --time=168:00:00 --ntasks-per-node=1 --gres=gpu:1 --mem=8000 --pty /bin/bash
cd /path/to/dl-master-voices/
export PRJ_PATH="${PWD}"
export PYTHONPATH=$PRJ_PATH
source mvenv/bin/activate
module load ffmpeg/intel/3.2.2
module load cuda/10.0.130
module load cudnn/10.0v7.6.2.24

python type/your/script/here param1 param2
``` 

### Running scripts in sbatch mode

``` 
sbatch ./sbatch/train_verifier.sbatch
``` 

The sbatch folder contains a file for each routine. 

### Running Jupyter notebooks

Run the notebook on HPC:
``` 
sbatch ./notebooks/run_jupyterlab_cpu.sbatch
``` 

Open the file .slurm file automatically created in ./notebooks and look for a line similar to the following to get the PORT and the LINK:
``` 
...
```
 
Open a terminal locally in your laptop and run:
``` 
ssh -L PORT:localhost:PORT NYUID@prince.hpc.nyu.edu
``` 

Open your browser and paste the LINK retrieved above:
``` 
``` 

## Using

### Train a speaker verifier
``` 
python -u ./routines/verifier/train.py --net "xvector" --learning_rate 0.001 --aggregation 'gvlad' --batch 32 --decay_step 15
```

### Test a speaker verifier
``` 
python -u ./routines/verifier/test.py --net "xvector/v000" --aggregation 'gvlad' --test_n_pair 1000
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


