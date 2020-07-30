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
unzip /beegfs/mm11333/data/data_20200706.zip
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

### Speaker Verification

#### Train
``` 
> python3 ./routines/verifier/train.py  --net "xvector" --val_n_pair 10000 
```

This script will save the model in ```./data/pt_models/xvector/v000/model.h5```.  

#### Test
``` 
> python3 -u ./routines/verifier/test.py --net "xvector/v000"
```

This script will test the model on Vox1-Test and finally save a CSV file with a trial pair and a 
similarity score per row in ```./data/pt_models/xvector/v000/test_vox1_sv_test.csv```.  

#### Pretrained Models

Pretrained models can be downloaded from [here](https://drive.google.com/drive/folders/15_ElEam7brk6lPheVV0fVXDQ2i3W_Iw5?usp=sharing).

|                  | status |     eer | thr@eer | thr@far1 | frr@far1 | no-trials |   loss |    acc |
|-----------------:|-------:|--------:|--------:|---------:|---------:|----------:|-------:|-------:|
|    resnet34/v000 |    run |  8.4464 |  0.8104 |   0.8629 |  32.7041 |     37720 | 1.7815 | 0.7809 |
|    resnet34/v001 |    run |  9.4989 |  0.7724 |   0.8385 |  35.5673 |     37720 | 3.9065 | 0.5765 |
|    resnet50/v000 |    run |  6.2195 |  0.7387 |   0.8118 |  25.9862 |     37720 | 0.6100 | 0.9616 |
|    resnet50/v001 |    run |  5.7688 |  0.7787 |   0.8436 |  21.5536 |     37720 | 2.3848 | 0.6683 |
| thin_resnet/v000 |    run |  6.2328 |  0.7559 |   0.8094 |  21.6914 |     37720 | 1.0338 | 0.9414 |
| thin_resnet/v001 |    run | 10.8059 |  0.7924 |   0.8719 |  41.5695 |     37720 | 4.2113 | 0.4495 |
|      vggvox/v000 |   stop | 10.7105 |  0.7095 |   0.8093 |  43.2291 |     37720 | 1.2410 | 0.8295 |
|      vggvox/v001 |    run |  8.1230 |  0.6839 |   0.7883 |  33.6957 |     37720 | 3.0182 | 0.5633 |
|     xvector/v000 |   stop | 12.5133 |  0.4682 |   0.6128 |  41.9512 |     37720 | 0.1421 | 0.9923 |
|     xvector/v001 |    run |  8.7778 |  0.8587 |   0.8953 |  30.6840 |     37720 | 0.9885 | 0.8660 |

Detailed information on ASVs performance can be computed within the ```speaker_verifier.ipynb``` notebook.  

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

**Step 1.** Create a folder for your master voice set in ```./data/vs_mv_data```. 

``` 
> mkdir -p ./data/vs_mv_data/specgan_m-m_sv/v000
``` 

**Step 2.** Copy the audio waveforms belonging to your master voice set in the new folder. 

**Step 3.** Run a routine to create a csv of trial pairs against a master voice.

``` 
> python3 routines/mv/create_pairs.py --mvset "specgan_m-m_sv/v000" 
``` 

This script creates a folder ```./data/vs_mv_pairs/mv/specgan_m-m_sv/v000``` with trials pairs for 
all the audio waveforms in the target master voice set. Specifically, for each audio waveform, this
script creates a csv file into the above folder, including trials pairs for each user belonging to the
Vox2-Master-Voice-Analysis set (columns: label, uservoice_path, mastervoice_path, gender).  

**Step 4.** Run a routine to create a csv of trial pairs with similarity scores. 

``` 
> python3 routines/mv/test_pairs.py --net "xvector/v000" --mvset "specgan_m-m_sv/v000" --policy "any"
``` 

This script creates a folder ```./data/pt_models/xvector/v000/mvcmp-any/```. For each csv file in
```./data/vs_mv_pairs/mv/specgan_m-m_sv/v000```, this script computes the similarity scores returned 
by ```xvector/v000``` for all the trial pairs in that csv. Finally, a copy of the csv file with an additional
column including the computed similarity scores is saved into the folder ```mvcmp-any``` (columns: 
label, uservoice_path, mastervoice_path, similarity_score, gender)

*NOTICE* This step should be adapted to support also the avg policy. 

**Step 5.** Open the notebook ```./notebooks/speaker_verifier.ipynb``` to inspect speaker verifiers' performance in terms of Equal Error Rate and Impersonation Rate.  

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


