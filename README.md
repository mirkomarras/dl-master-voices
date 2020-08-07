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
wget "<link>"
unzip data_20200807.zip
rm -r data_20200807.zip
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

## Usage (Command Line)

### Speaker Modelling

Speaker models aim to provide compact 1-D floating-point representations (i.e., embeddings) of vocal audio files, 
so that the embeddings extracted from vocal audio files of the same speaker are similar to each other (high
intra-class similarity) and those extracted from vocal audio files of different speakers are very dissimilar to
each other (low inter-class similarity). Depending on the architecture, a speaker model can take directly the raw
audio, the audio spectrogram, or the audio filterbank (see [here](https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html) 
for a more detailed discussion).  

With this repository, a range of pretrained models are available and can be downloaded from
[here](https://drive.google.com/drive/folders/15_ElEam7brk6lPheVV0fVXDQ2i3W_Iw5?usp=sharing). Each model 
should be copied into the appropriate sub-folder in ```./data/vs_mv_models```. The best model performance 
on the trial verification pairs provided with the VoxCeleb1-Test dataset are reported below. 

|                  | input  |     eer | thr@eer | thr@far1 | frr@far1 | 
|-----------------:|-------:|--------:|--------:|---------:|---------:|
|    resnet34/v002 |   spec |   6.763 |  0.8488 |   0.8834 |  24.0244 | 
|    resnet34/v003 |   spec |   8.207 |  0.7161 |   0.7982 |  31.8823 | 
|    resnet50/v002 |   spec |   6.182 |  0.7395 |   0.8110 |  25.6734 |
|    resnet50/v003 |   spec |   5.015 |  0.7721 |   0.8277 |  17.6193 |
| thin_resnet/v002 |   spec |   5.570 |  0.7700 |   0.8159 |  18.4783 |
| thin_resnet/v003 |   spec |   9.310 |  0.7607 |   0.8411 |  36.4104 |
|      vggvox/v002 |   spec |  10.710 |  0.7095 |   0.8093 |  43.2291 |
|      vggvox/v003 |   spec |   6.932 |  0.7625 |   0.8292 |  27.6087 |
|     xvector/v002 |   filt |  12.513 |  0.4682 |   0.6128 |  41.9512 |
|     xvector/v003 |   filt |   8.245 |  0.8430 |   0.8817 |  28.2503 |

#### Train

``` 
python3 ./routines/verifier/train.py  --net "xvector" --val_n_pair 10000 
```

This script will save the model in ```./data/vs_mv_models/xvector/v000/model.h5```.  

#### Test
``` 
> python3 -u ./routines/verifier/test.py --net "xvector/v000"
```

This script will test the model on Vox1-Test and finally save a CSV file with a trial pair and a 
similarity score per row in ```./data/vs_mv_models/xvector/v000/test_vox1_sv_test.csv```.  

#### Pretrained Models


### Spectrogram Generation (GAN)

#### Train

```
> python3 routines/gan/train.py -d voxceleb-male -e 200
``` 

#### Test (show samples)
``` 
> python3 routines/gan/preview.py -d voxceleb-male
``` 

### Master Voice Optimization

#### Generation
...

#### Test

**Step 1.** Create a folder for your master voice set in ```./data/vs_mv_data```. For instance:

``` 
> mkdir -p ./data/vs_mv_data/real_f-f_sv/v000
``` 

**Step 2.** Copy the audio waveforms belonging to your master voice set in the new folder. 

**Step 3.** Run a routine to create a csv of trial pairs against a master voice.

``` 
> python3 routines/mv/create_pairs.py --mv_set "real_f-f_sv/v000" 
``` 

This script creates a folder ```./data/vs_mv_pairs/mv/real_f-f_sv/v000``` with trials pairs for 
all the audio waveforms in the target master voice set. Specifically, for each audio waveform, this
script creates a csv file into the above folder, including trials pairs for each user belonging to the
Vox2-Master-Voice-Analysis set (columns: label, path1, path2, gender).  

If ```--mv_set``` is not specified, this script creates a csv file for each set in ```./data/vs_mv_pairs/mv```.

**Step 4.** Create a noise folder
```
> mkdir ./data/vs_noise_data
```

**Step 5.** Run a routine to test all the master voice sets against the target verifier. 

``` 
> python3 routines/mv/test_pairs.py --net "xvector/v000" 
``` 

This script creates a folder ```./data/vs_mv_models/xvector/v000/mvcmp_any/```. For each csv file in
```./data/vs_mv_pairs/mv/```, this script computes the similarity scores returned by ```xvector/v000``` 
for each trial pair in the current csv. Finally, a copy of the csv file with an additional
column that includes the computed similarity scores is saved into the folder ```mvcmp_any``` (columns: 
score, label, path1, path2, gender). 

**Step 5.** Open the notebook ```./notebooks/speaker_verifier.ipynb``` to inspect speaker verifiers' performance in terms of Equal Error Rate and Impersonation Rate. This notebook will use all the csv files generated above. 

## Usage (APIs)

...

## NYU HPC


### Running scripts in interactive mode

``` 
srun --time=168:00:00 --ntasks-per-node=1 --gres=gpu:1 --mem=64000 --pty /bin/bash
export PRJ_PATH="${PWD}"
export PYTHONPATH=$PRJ_PATH
source mvenv/bin/activate
module load ffmpeg/intel/3.2.2
module load cuda/10.0.130
module load cudnn/10.0v7.6.2.24
python -u ./routines/verifier/train.py --net "resnet50vox" --learning_rat 0.001 --batch 32 --augment 0 


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

## Contributing to the Code

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


