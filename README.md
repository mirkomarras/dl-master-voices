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

This repository contains the source code of the following articles:

- "**Adversarial Optimization for Dictionary Attacks on Speaker Verification**", **INTERSPEECH 2019**, [BibTex](https://dblp.uni-trier.de/rec/bibtex/conf/interspeech/MarrasKMF19). 
Full text available on [ISCA DL (open)](https://www.isca-speech.org/archive/Interspeech_2019/pdfs/2430.pdf) and poster 
available on [GDrive](https://drive.google.com/file/d/1ORlboRh7i1Oi14vLdrGxGuY-iqg8Ct5Z/view?usp=sharing).

## Table of Contents
- [Installation](#installation)
- [Data Folder Description](#data-folder-description) 
- [Usage (Command Line)](#usage-command-line)
    - [Speaker Modelling](#speaker-modelling)
        - [Train](#train)
        - [Test](#test)
    - [Spectrogram Generation (GAN)](#spectrogram-generation-gan)
        - [Train](#train-1)
        - [Test](#test-show-samples)
    - [Master Voice Optimization](#master-voice-optimization)
        - [Generation](#generation)
        - [Test](#test-1)
        - [Analysis](#analysis)
- [Usage (APIs)](#usage-apis)
- [Usage (NYU HPC)](#usage-nyu-hpc)
    - [Running Scripts in Interactive Mode](#running-scripts-in-interactive-mode)
    - [Running Scripts in Batch Mode](#running-scripts-in-batch-mode)
    - [Running Jupyter Notebooks](#running-jupyer-notebooks)
- [Contributing](#contributing)
- [Citations](#citations)
- [License](#license)

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

Copy the data folder (around 5GB):
``` 
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1PAM7yaDMjQMCndLBUPBkXXqHG9k6HXa_' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1PAM7yaDMjQMCndLBUPBkXXqHG9k6HXa_" -O data_20200807.zip && rm -rf /tmp/cookies.txt
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

## Data Folder Description 

``` 
data
├── voxceleb1 -> /beegfs/mm11333/data/voxceleb1
├── voxceleb2 -> /beegfs/mm11333/data/voxceleb2
├── vs_mv_data (sets of master voices)
│   ├── vggvox-v000_real_f-f_mv
│   ├── vggvox-v000_real_f-f_sv
│   ├── vggvox-v000_real_m-m_mv
│   ├── vggvox-v000_real_m-m_sv
│   ├── vggvox-v000_wavegan-v000_f-f_mv
│   ├── vggvox-v000_wavegan-v000_f-f_sv
│   ├── vggvox-v000_wavegan-v000_m-m_mv
│   ├── vggvox-v000_wavegan-v000_m-m_sv
│   ├── vggvox-v000_wavegan-v000_n-f_mv
│   ├── vggvox-v000_wavegan-v000_n-f_sv
│   ├── vggvox-v000_wavegan-v000_n-m_mv
│   └── vggvox-v000_wavegan-v000_n-m_sv
├── vs_mv_models (sets of pre-trained speaker models)
│   ├── ms-gan
│   ├── resnet34
│   ├── resnet50
│   ├── thin_resnet
│   ├── vggvox
│   └── xvector
├── vs_mv_pairs (set of utility data)
│   ├── data_mv_vox2_all.npz (train-test splits for master voice analysis in VoxCeleb2-Dev)
│   ├── meta_data_vox12_all.csv (id-gender correspondence for VoxCeleb1-2 users)
│   ├── mv (folder that includes csv files with trial verification pairs for master voices)
│   ├── trial_pairs_vox1_test.csv (trial verification pairs from VoxCeleb1-Test)
│   └── trial_pairs_vox2_test.csv (trial verification pairs from VoxCeleb2-Dev MV-Train)
│   ├── trial_pairs_vox2_mv.csv (paths to enrolled templates for users in VoxCeleb2-Dev MV-Test)
└── vs_noise_data (sets of background noise files for playback-n-recording)
    ├── general
    ├── microphone
    ├── room
    └── speaker
``` 

## Usage (Command Line)

### Speaker Modelling

Speaker models aim to provide compact 1-D floating-point representations (i.e., embeddings) of vocal audio files, 
so that the embeddings extracted from vocal audio files of the same speaker are similar to each other (high
intra-class similarity) and those extracted from vocal audio files of different speakers are very dissimilar to
each other (low inter-class similarity). Depending on the architecture, a speaker model can take as an input 
directly the raw audio, the audio spectrogram, or the audio filterbank (see [here](https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html) 
for a more detailed discussion).  

With this repository, a range of pretrained models are available and can be downloaded from
[here](https://drive.google.com/drive/folders/1KYmizReZ3HHd6qH9r8Hpu24iHVXd8Ikb?usp=sharing). Each model 
should be copied into the appropriate sub-folder in ```./data/vs_mv_models```. The best model performance 
on the verification pairs provided with the VoxCeleb1-Test dataset are reported below. 

|   Model ID       | Input  | Shape          |   Size (MB)  |     EER | THR@EER | THR@FAR1% | FRR@FAR1% | 
|-----------------:|-------:|---------------:|-------------:|--------:|--------:|----------:|----------:|
|    resnet34/v002 |   spec |   (256,None,1) |    360       |   6.763 |  0.8488 |   0.8834 |  24.0244 | 
|    resnet34/v003 |   spec |   (256,None,1) |    360       |   8.207 |  0.7161 |   0.7982 |  31.8823 | 
|    resnet50/v002 |   spec |   (256,None,1) |    427       |   6.182 |  0.7395 |   0.8110 |  25.6734 |
|    resnet50/v003 |   spec |   (256,None,1) |    427       |   5.015 |  0.7721 |   0.8277 |  17.6193 |
| thin_resnet/v002 |   spec |   (256,None,1) |    128       |   5.570 |  0.7700 |   0.8159 |  18.4783 |
| thin_resnet/v003 |   spec |   (256,None,1) |    128       |   9.310 |  0.7607 |   0.8411 |  36.4104 |
|      vggvox/v002 |   spec |   (256,None,1) |    100       |  10.710 |  0.7095 |   0.8093 |  43.2291 |
|      vggvox/v003 |   spec |   (256,None,1) |    100       |   6.932 |  0.7625 |   0.8292 |  27.6087 |
|     xvector/v002 |   filt |   (None, 24)   |    104       |  12.513 |  0.4682 |   0.6128 |  41.9512 |
|     xvector/v003 |   filt |   (None, 24)   |    104       |   8.245 |  0.8430 |   0.8817 |  28.2503 |

#### Train

This toolbox allows you to train a speaker model from scratch. For instance, a x-vector model can be trained
by running the following script and indicating that specific type of speaker model to train. 

``` 
python3 ./routines/verifier/train.py  --net "xvector"
```

The training script will save in ```./data/vs_mv_models/{net}/v{xxx}/```: 
- the model ```model.h5```;
- the training history per epoch ```history.csv``` (loss, acc, err, far@eer, frr@eer, thr@eer, far@far1, frr@far1, thr@far);
- the training parameters in ```params.txt``` (line: key, value). 

To resume the training of an existing model, the model version must be specified (e.g., ``` --net "xvector/v000"``` ).

The training script can be configured in order to train different types of models with proper parameters. The
most common parameters that can be customized are provided below.

``` 
--net 'x-vector'                   (Model in ['x-vector','vggvox','thin_resnet','resnet34','resnet50'])
--audio_dir './data/voxceleb1/dev' (Directories with wav training files)
--n_epochs 1024                    (Number of training epochs)
--batch 64                         (Size of the training batches)
--learning_rate 0.001              (Starting learning rate)
--loss 'softmax'                   (Type of loss in ['softmax','amsoftmax'])
--aggregation 'gvlad'              (Type of aggregation in ['avg','vlad','gvlad'])
--val_n_pair 1000                  (Number of validation trials pairs)
--n_seconds 3                      (Training audio lenght in seconds)
``` 

#### Test
This toolbox provides a script to compute the performance of a speaker model. For instance, to test a pre-trained
x-vector model, the following command should be run. Check that the following command returns ```EER 8.245```.  

``` 
python3 -u ./routines/verifier/test.py --net "xvector/v003"
```

By default, this script will test the model on the standard VoxCeleb1 trial verification pairs provided together
with the original dataset (37,720 pairs). The CSV file with the similarity scores returned by the model for each
trial pair will be saved in ```./data/vs_mv_models/{net}/v{xxx}/scores_vox1_test.csv``` (similarity,label). 
  
The current speaker model can be tested also on a set of trial verification pairs of the VoxCeleb2-Dev part devoted
to master voice training (37,720 randomly-generated pairs). To do this, the following two parameters must be specified:
```--test_base_path "./data/voxceleb2/dev" --test_pair_path "./data/vs_mv_pairs/trial_pairs_vox2_test.csv"```.
The resulting labels and similarity scores will be saved in ```./data/vs_mv_models/{net}/v{xxx}/scores_vox2_test.csv```.

### Spectrogram Generation (GAN)
The GAN models included into this toolbox can generate fake spectrograms, from 1-D latent vectors of size 128. This kind 
of models is particularly useful to search master voice in a latent space and not in a known population. With this repository,
a range of GAN models are available and can be downloaded from [here](https://drive.google.com/drive/folders/1NMGtukTPKod7T7fHp47kGgv7QVcVxIXC?usp=sharing). 
Each model should be copied into the appropriate sub-folder in ```./data/vs_mv_models```. Some details on the pre-trained
GAN models are provided below. 

|              GAN ID |  Input |      Output | Comments                  |
|--------------------:|-------:|------------:|---------------------------|
|  ms-gan/female/v000 | (128,) | (256,256,1) | Normalized spectrograms   |
|  ms-gan/female/v001 | (128,) | (256,256,1) | Unnormalized spectrograms |
|    ms-gan/male/v000 | (128,) | (256,256,1) | Normalized spectrograms   |
|    ms-gan/male/v001 | (128,) | (256,256,1) | Unnormalized spectrograms |
| ms-gan/neutral/v000 | (128,) | (256,256,1) | Normalized spectrograms   |
| ms-gan/neutral/v001 | (128,) | (256,256,1) | Unnormalized spectrograms |

#### Train
This toolbox allows you to train a GAN model from scratch. For instance, a Multi-Scale GAN model optimized for male voices
can be trained by running the following script and indicating that specific type of speaker model to train. 

``` 
python3 ./routines/gan/train.py --model "ms-gan" --gender "male"
```

The training script will save in ```./data/vs_mv_models/{net}/v{xxx}/{gender}```: 
- the generator model ```generator.h5```;
- the discriminator model ```discriminator.h5```;
- the preview outputs every 10 steps ```preview_{step}.jpg```;
- the loss and accuracy progress ```progress.jpg```;
- the statistics on the training ```stats.json``` (i.e., losses, accuracies, history (fake, real), model type, args). 

To resume the training of an existing model, the model version must be specified (e.g., ``` --net "ms-gan/v000"```).

The training script can be configured in order to train different types of models with proper parameters. The
most common parameters that can be customized are provided below.

``` 
--model 'ms-gan'                   (Model in ['ms-gan','dc-gan'])
--dataset './data/voxceleb1/dev'   (Directories with wav training files)
--gender 'female'                  (Gender in ['female', 'male', 'neutral'])
--length 2.58                      (Time length of the generated spectrograms)
--batch 64                         (Size of the training batches)
``` 

#### Test (show samples)
This toolbox provides a script to compute a set of randomly-generated spectrograms from a pre-trained GAN. For instance, to 
preview samples from a pre-trained MultiScale GAN model optimized for male voices, the following command should be run.   

``` 
python3 -u ./routines/gan/preview.py --model "ms-gan" --gender "male" --version 1
```

By default, this script will show the preview spectrograms. If you are interested in directly creating audio examples from 
the spectrograms generated by a GAN, the following script should be run. 

``` 
python3 -u ./routines/gan/griffin_lim_preview.py --model "ms-gan" --gender "male" --version 1
```

This script will create randomly-generated audios by inverting the fake spectrograms through the Griffin-Lim algorithm. 
The audio files will be saved in ```./data/vs_mv_models/{net}/v{xxx}/{gender}/gla_samples```. 

### Master Voice Optimization

Master voices are defined as a family of adversarial audio files which match large populations of speakers 
by chance with high probability. This toolbox organizes master voices in sets according to the speaker model and 
the seed voices used for optimization. With this repository, a range of seed and master voice sets are available 
and can be downloaded from [here](https://drive.google.com/drive/folders/1gQeWT7kI6eYXIBoVg4yTkJgw5UdDt0BW?usp=sharing). 
Each set should be copied into the appropriate sub-folder in ```./data/vs_mv_data```. Some details on the current
master voice sets are provided below. 

|                     Set ID      | Number of Samples |                                          Comments |
|--------------------------------:|------------------:|--------------------------------------------------:|
|    vggvox-v000_real_f-f_mv      |                50 | Uniformly sampled based on the false accepts |
|    vggvox-v000_real_m-m_mv      |                50 | Uniformly sampled based on the false accepts |
| vggvox-v000_wavegan-v000_f-f_mv |                 5 |                                                   |
| vggvox-v000_wavegan-v000_m-m_mv |                 5 |                                                   |
| vggvox-v000_wavegan-v000_n-f_mv |                 5 |                                                   |
| vggvox-v000_wavegan-v000_n-m_mv |                 5 |                                                   |
|    vggvox-v000_real_f-f_sv      |                50 | Uniformly sampled based on the false accepts |
|    vggvox-v000_real_m-m_sv      |                50 | Uniformly sampled based on the false accepts |
| vggvox-v000_wavegan-v000_f-f_sv |                 5 |                                                   |
| vggvox-v000_wavegan-v000_m-m_sv |                 5 |                                                   |
| vggvox-v000_wavegan-v000_n-f_sv |                 5 |                                                   |
| vggvox-v000_wavegan-v000_n-m_sv |                 5  |                                                   |

Each set is named with the following convention: ```{netv-vxxx}_{netg-vxxx|real}_{seed_gender}-{opt_gender}_{sv|mv}```,
where ```netv-vxxx``` are the speaker model and its version; ```netg-vxx``` are the gan model and its version; 
```real``` is a name for non-gan-generated sets; ```seed_gender``` is the gender against which the gan has been
trained or, in general, the gender of the individuals in the audio files (f:female, m:male, n:neutral); ```opt_gender```
is the gender against which the seed voice has been optimized; ```sv``` indicates seed voice sets; ```mv``` indicates 
their corresponding master voice sets.

#### Generation
This toolbox includes three main ways of generating master voices:

1. Optimize an individual seed voice: 
    ``` 
    python -u ./routines/mv/train.py --netv "vggvox/v003" --seed_voice "./tests/original_audio.wav" 
    ``` 
    This command will save seed/master voices in ```{netv-vxxx}_{real}_{opt_gender}_{sv|mv}```. 
    
2. Optimize a set of seed voices: 
    ``` 
    python -u ./routines/mv/train.py --netv "vggvox/v003" --seed_voice "vggvox-v000_real_f-f_mv/v000"
    ``` 
    This command will save seed/master voices in ```{netv-vxxx}_{real}_{opt_gender}_{sv|mv}```.
    
3. Optimize a set of gan-generated voices: 
    ``` 
    python -u ./routines/mv/train.py --netv "vggvox/v003" --netg "ms-gan/v001"
    ``` 
    This command will save seed/master voices in ```{netv-vxxx}_{netg-vxxx}_{seed_gender}-{opt_gender}_{sv|mv}```.

For each master voice, the following files will be saved (we provide an example for a sample_0 master voice):
- the master voice file ```sample_0.wav```;
- the master voice spectrogram/latent-vector ```sample_0.npy```;
- the master voice optimization history ```sample_0.hist``` (list of training impersonation rates at EER and FAR1% thrs). 

The optimization script can be configured in order to optimize different types of master voices with proper parameters. 
The most common parameters that can be customized are provided below.

``` 
--netg_gender 'female'             (Peculiar gender of the GAN in ['neutral','female','male'])
--mv_gender 'female'               (Gender of optimization audio files in ['neutral','female','male'])
--n_examples 100                   (Number of master voices to generate - only for GAN-based processes)
--n_epochs 1024                    (Number of optimization epochs)
--batch 64                         (Size of the optimization batches)
--learning_rate 0.001              (Starting learning rate for optimization)
--n_templates 10                   (Number of enrolled templates per user to test impersonation)
--n_seconds 3                      (Optimization audio lenght in seconds)
``` 

#### Test

This toolbox provides a script to compute the similarity scores between the audio files belonging to the master voice
sets and the audio files belonging to the enrolled templates of users in the master-voice analysis part of VoxCeleb2-Dev.
These scores will be then used to compute the impersonation rates of each master voice in the considered sets. To this
end, the following script will scan all the master voice sets in ```./data/vs_mv_data``` and compute the similarity
scores for each voice in those sets, given a certain speaker model (already processed sets will be skipped). 

This toolbox includes two verification policy, which influence the way the similarity scores are computed and saved:
- ```any```: the similarity score for each enrolled user's template and master voice is computed;
- ```avg```: the embeddings of the user's templates are averaged and a unique similarity score per user is saved. 

For instance, to compute similarity scores from a pre-trained xvector model, the following command should be run: 

``` 
python3 routines/mv/test.py --net "vggvox/v003" 
``` 

This script will compute similarity scores for both the policies, with 10 templates per user. First, two sub-folders 
that include all the csv files with the testing results are created in ```./data/vs_mv-models/{net}/{vxxx}```, namely
```mvcmp_any``` for the any policy and ```mvcmp_avg``` for the avg policy. Then, for each audio in the master voice 
sets saved in ```./data/vs_mv_data```, this scripts creates a csv file that includes the trial verification pair results
(columns: score, path1, path2, gender), obtained by computing the similarity scores between the current master 
voice and the audio files belonging to the enrolled templates of users in the master-voice analysis part of 
VoxCeleb2-Dev. For the any policy, by default, ten rows per user are saved in each csv file. For the avg policy, one 
row per user is saved in each csv file. 

To test multiple speaker models at the same time, you can specify more than one model in the ```-net``` parameter 
(e.g., ```--net "vggvox/v003,xvector/v003" ``` ). 

#### Analysis
This toolbox is accompanied by a notebook ```./notebooks/speaker_verifier.ipynb``` that includes the code needed to
test speaker model performance in terms of Equal Error Rate and Impersonation Rate. This notebook will use all the csv 
files generated as described in the Test section. 

## Usage (APIs)

...

## Usage (NYU HPC)


### Running scripts in interactive mode

``` 
srun --time=168:00:00 --ntasks-per-node=1 --gres=gpu:1 --mem=64000 --pty /bin/bash
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

## Contributing 

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


