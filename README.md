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
each other (low inter-class similarity). Depending on the architecture, a speaker model can take as an input 
directly the raw audio, the audio spectrogram, or the audio filterbank (see [here](https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html) 
for a more detailed discussion).  

With this repository, a range of pretrained models are available and can be downloaded from
[here](https://drive.google.com/drive/folders/1KYmizReZ3HHd6qH9r8Hpu24iHVXd8Ikb?usp=sharing). Each model 
should be copied into the appropriate sub-folder in ```./data/vs_mv_models```. The best model performance 
on the verification pairs provided with the VoxCeleb1-Test dataset are reported below. 

|                  | Input  | Shape          |   Size (MB)  |     EER | THR@EER | THR@FAR1% | FRR@FAR1% | 
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
--audio_dir "./data/voxceleb1/dev" (Directories with wav training files)
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

#### Train

```
> python3 routines/gan/train.py -d voxceleb-male -e 200
``` 

#### Test (show samples)
``` 
> python3 routines/gan/preview.py -d voxceleb-male
``` 

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
This toolbox includes three main ways of optimizing master voices:

1. Optimize an individual seed voice: 
    ``` 
    python -u ./routines/mv/train.py --netv "vggvox/v003" --seed_voice "./tests/original_audio.wav" 
    ``` 
    This command will save seed/master voices in ```{netv-vxxx}_{real}_{opt_gender}_{sv|mv}```. 
    
2. Optimize a set of seed voices: 
    ``` 
    python -u ./routines/mv/train.py --netv "vggvox/v003" --seed_voice "./data/vs_mv_data/vggvox-v000_real_f-f_mv/v000"
    ``` 
    This command will save seed/master voices in ```{netv-vxxx}_{real}_{opt_gender}_{sv|mv}```.
    
3. Optimize a set of gan--generated voices: 
    ``` 
    python -u ./routines/mv/train.py --netv "vggvox/v003" --netg "ms-gan/v001"
    ``` 
    This command will save seed/master voices in ```{netv-vxxx}_{netg-vxxx}_{seed_gender}-{opt_gender}_{sv|mv}```.

For each master voice, the following files will be saved (we provide an example for a sample_0 master voice):
- the master voice file ```sample_0.wav```;
- the master voice spectrogram/latent-vector ```sample_0.npy```;
- the master voice optimization history ```sample_0.hist``` (list of impersonation rates at EER and FAR1% thrs). 

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

**Step 1.** Create a folder for your master voice set in ```./data/vs_mv_data```. For instance:

``` 
python -u ./routines/mv/train.py --netv "vggvox/v003" --seed_voice "./tests/original_audio.wav" 
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


