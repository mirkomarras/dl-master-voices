# Master Voice Toolbox
[![Build Status](https://travis-ci.org/pages-themes/cayman.svg?branch=master)](https://travis-ci.org/pages-themes/cayman)
[![GitHub version](https://badge.fury.io/gh/boennemann%2Fbadges.svg)](http://badge.fury.io/gh/boennemann%2Fbadges)
[![Dependency Status](https://david-dm.org/boennemann/badges.svg)](https://david-dm.org/boennemann/badges)
[![Open Source Love](https://badges.frapsoft.com/os/gpl/gpl.svg?v=102)](https://github.com/ellerbrock/open-source-badge/)

[Mirko Marras](https://www.mirkomarras.com/)<sup>1</sup>, [Paweł Korus](http://kt.agh.edu.pl/~korus/)<sup>2,3</sup>, 
[Nasir Memon](https://engineering.nyu.edu/faculty/nasir-memon)<sup>2</sup>, [Gianni Fenu](http://people.unica.it/giannifenu/)<sup>1</sup>
<br/><sup>1</sup> University of Cagliari, <sup>2</sup> New York University, <sup>3</sup> AGH University of Science and Technology

A Python toolbox for creating and testing impersonation capabilities of a potentially large family of adversarial audio
samples called **Master Voices** (MVs) which match large populations of speakers by chance with high probability. 

## Table of Contents
- [Master Voice Toolbox](#master-voice-toolbox)
  * [Setup](#setup)
    + [Step 1: Install Python3](#step-1--install-python3)
    + [Step 2: Create a Virtual Environment](#step-2--create-a-virtual-environment)
    + [Step 3: Clone the Repository](#step-3--clone-the-repository)
    + [Step 4: Install Toolbox Requirements](#step-4--install-toolbox-requirements)
  * [Getting Started](#getting-started)
    + [Step 1: Download Source Data Sets](#step-1--download-source-data-sets)
    + [Step 2: Prepare a Speaker Verifier](#step-2--prepare-a-speaker-verifier)
      - [Step 2.1: Train](#step-21--train)
      - [Step 2.2: Evaluate](#step-22--evaluate)
      - [Step 2.3: Extract](#step-23--extract)
      - [Step 2.4: Use Pre-Trained Speaker Verifiers](#step-24--use-pre-trained-speaker-verifiers)
    + [Step 3: Train a Generative Adversarial Network (GAN)](#step-3--train-a-generative-adversarial-network--gan-)
      - [Step 3.1: Train](#step-31--train)
      - [Step 3.2: Use Pre-Trained GANs](#step-32--use-pre-trained-gans)
    + [Step 4: Generate and Evaluate a Master Voice](#step-4--generate-and-evaluate-a-master-voice)
      - [Step 4.1: Train by Spectrogram Changes](#step-41--train-by-spectrogram-changes)
      - [Step 4.2: Train by GAN](#step-42--train-by-gan)
      - [Step 4.3: Evaluate](#step-43--evaluate)
      - [Step 4.4: Use Pre-Computed Master Voice Sets](#step-44--use-pre-computed-master-voice-sets)
  * [Contribution](#contribution)
  * [Citations](#citations)
  * [License](#license)
  
## Setup 

### Step 1: Install Python3
```
$ sudo apt-get update
$ sudo apt-get install python3.5
```

### Step 2: Create a Virtual Environment
```
$ python3 -m virtualenv venv
$ source venv/bin/activate
```

### Step 3: Clone the Repository
```
$ git clone https://github.com/mirkomarras/dl-master-voices.git
$ export PYTHONPATH=/path/to/dl-master-voices/
```

### Step 4: Install Toolbox Requirements
```
$ pip install -r dl-master-voices/requirements.txt
```

## Getting Started

### Step 1: Download Source Data Sets
[VoxCeleb1](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html) contains over 100,000 utterances for 1,251 
celebrities, extracted from videos uploaded to YouTube. 

[VoxCeleb2](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html) contains over 1 million utterances for 6,112 
celebrities, extracted from videos uploaded to YouTube.

### Step 2: Prepare a Speaker Verifier

#### Step 2.1: Train
The training script looks for utterances in *data_source_vox1* and *data_source_vox2* folders, excluding the utterances 
from users involved in master voice analysis. Both data sources must point to a folder including *dev* and *test* subfolders. 

The available verifiers are *xvector*, *vggvox*, *resnet34vox*, and *resnet50vox*. By default, xvector models
are trained on *300x24*-sized filterbanks and return vectors of size *1024*, while the other models are trained on 
*512x300x1*-sized spectrograms and return vectors of size *512*. Each model is trained for *n_epochs=40* on *batch_size=32* with *learning_rate=1e-1*.
The script saves the pre-trained model into the folder *model_dir*.

Voice detection (*vad=[True|False]*) and augmentation (*aug=[0:no|1:aug_any|2:aug_seq|3:aug_prob]*) can be performed. 

A sample training command is provided below: 

```
$ python ./code/train_speaker_verificator.py 
  --verifier "xvector"
  --data_source_vox1 "/beegfs/mm10572/voxceleb1" 
  --data_source_vox2 "/beegfs/mm10572/voxceleb2" 
  --noises_dir "./data/noise"
  --model_dir "./models/xvector/model"
  --n_epochs 1024
  --batch_size 32
  --learning_rate 1e-1
  --shuffle True
  --dropout_proportion 0.1
  --sample_rate 16000
  --preemphasis 0.97
  --frame_stride 0.01
  --frame_size 0.025
  --num_fft 512
  --min_chunk_size 10
  --max_chunk_size 300
  --aug 3
  --vad False
  --prefilter True
  --normalize True
  --nfilt 24
  --print_interval 1
```

#### Step 2.2: Evaluate
The testing script looks for trial pairs in *trials_file* on the dataset *data_source*. Such pairs are compared by the
*verifier* available at *model_dir* through a *comparison_metric=["euclidean_dist","cosine_similarity"]*. 

The available verifiers are *xvector*, *vggvox*, *resnet34vox*, and *resnet50vox*. By default, xvector models
are tested on *300x24*-sized filterbanks, while the other models are tested on *512x300x1*-sized spectrograms. The script 
saves the comparison results into the file *result_file*.

Voice detection (*vad=[True|False]*) and augmentation (*aug=[0:no|1:aug_any|2:aug_seq|3:aug_prob]*) can be performed. 

A sample testing command is provided below: 

```
$ python ./code/test_speaker_verificator.py
  --verifier "xvector"
  --data_source "/beegfs/mm10572/voxceleb1/test"
  --noises_dir "./data/noise"
  --model_dir "./models/xvector/model"
  --result_file "./data/vox1_eer/result_pairs_voxceleb1.csv"
  --trials_file "./data/vox1_eer/trial_pairs_voxceleb1.csv"
  --comparison_metric "euclidean_dist"
  --sample_rate 16000
  --preemphasis 0.97
  --frame_stride 0.01
  --frame_size 0.025
  --num_fft 512
  --min_chunk_size 10
  --max_chunk_size 300
  --aug 0
  --vad False
  --prefilter True
  --normalize True
  --nfilt 24
```

#### Step 2.3: Extract
The extraction script computes embeddings for files in *test_paths* on the dataset *data_source*. Such *embs_size*-dim vectors are extracted by the
*verifier* available at *model_dir* through a *comparison_metric=["euclidean_dist","cosine_similarity"]*. 

The available verifiers are *xvector*, *vggvox*, *resnet34vox*, and *resnet50vox*. By default, xvectors
are extracted from *300x24*-sized filterbanks, while the other vectors are extracted from *512x300x1*-sized spectrograms. The script 
saves the embeddings into the file *embs_path*.

Voice detection (*vad=[True|False]*) and augmentation (*aug=[0:no|1:aug_any|2:aug_seq|3:aug_prob]*) can be performed. 

A sample extraction command is provided below: 

``` 
$ python ./code/extract_speaker_verificator.py
  --verifier "xvector"
  --data_source "/beegfs/mm10572"
  --model_dir "./models/xvector/model"
  --embs_path "./data/vox2_embs/embs_xvector.csv"
  --embs_size 512
  --test_paths "./data/vox2_mv/test_vox2_abspaths_1000_users"
  --noises_dir "./data/noise"
  --start_index 0
  --sample_rate 16000
  --preemphasis 0.97
  --frame_stride 0.01
  --frame_size 0.025
  --num_fft 512
  --min_chunk_size 10
  --max_chunk_size 300
  --aug 0
  --vad False
  --prefilter True
  --normalize True
  --nfilt 24
```

#### Step 2.4: Use Pre-Trained Speaker Verifiers
Please find the resulting pre-trained models in the table below.
 
| Name | Pre-Trained Model | Equal Error Rate  |
|:----:|:----------------:|:----------------:|
|  X-Vector    |       [Link]()           |        |        
|  VGGVox-Vector    |      [Link]()              |    |            
|  ResNet34Vox-Vector    |       [Link]()             |   |             
|  ResNet50Vox-Vector    |      [Link]()              |    |            

### Step 3: Train a Generative Adversarial Network (GAN)

#### Step 3.1: Train

#### Step 3.2: Use Pre-Trained GANs
Please find the resulting pre-trained models in the table below.
 
| Name | Seed | Pre-Trained Model | 
|:----:|:----------------:|:----------------:|
|  WaveGAN    | M+F |      [Link]()           |         
|  WaveGAN   |   M |   [Link]()              |              
|  WaveGAN    |  F |     [Link]()             |        

### Step 4: Generate and Evaluate a Master Voice

#### Step 4.1: Train by Spectrogram Changes

#### Step 4.2: Train by GAN
The training script computes master voices saved in *gan_mv_base_path* and optimized for the utterances *train_paths*.. 
The master voices are generated by the GAN available at *gan_metafile_path*. The available verifiers are *xvector*, 
*vggvox*, *resnet34vox*, and *resnet50vox*. 

Voice detection (*vad=[True|False]*) and augmentation (*aug=[0:no|1:aug_any|2:aug_seq|3:aug_prob]*) can be tested. 

A sample training command is provided below: 

``` 
$ python ./code/train_mv_by_gan.py
  --verifier "vggvox"
  --model_dir "./models/vggvox/ks_model/vggvox.h5"
  --noises_dir "./data/noise"
  --speaker_ir "./data/noise/ir_speaker/IR_ClestionBD300.wav"
  --room_ir "./data/noise/ir_room/BRIR.wav"
  --mic_ir "./data/noise/ir_mic/IR_OktavaMD57.wav"
  --post_processing True
  --meta_file "./data/vox2_meta/meta_vox2.csv"
  --meta_gender_col "gender"
  --meta_gender_male "m"
  --train_paths "./data/vox2_mv/train_vox2_abspaths_1000_users"
  --train_labels "./data/vox2_mv/train_vox2_labels_1000_users"
  --train_embs "./data/vox2_mv/train_vox2_embs_1000_users.npy"
  --gan_metafile_path "./models/stdgan/infer.meta"
  --gan_ckpt_path "./models/stdgan/model.ckpt-62552"
  --gan_mv_base_path "./sets/sample-set"
  --gan_ov_base_placeholder "audio_original_"
  --gan_mv_base_placeholder "gan_mv_base_placeholder"
  --batch_size 128
  --n_iterations 1000
  --learning_rate 1e-2
  --min_similarity 0.25
  --max_similarity 1.00
  --utterance_type "female"
  --attempts 50
  --params_file_path "params_file_path"
  --thr_eer 0.53
  --thr_far1 0.74
  --sample_rate 16000
  --preemphasis 0.97
  --frame_stride 0.01
  --frame_size 0.025
  --num_fft 512
  --min_chunk_size 10
  --max_chunk_size 300
  --aug 0
  --vad False
  --prefilter True
  --normalize True
  --nfilt 24
``` 

#### Step 4.3: Evaluate
The evaluation script computes impersonation rates for master voices in *test_mv_set* on the dataset *data_source* via
the *test_paths* represented by *test_embs* and *test_labels*. Such rates are computed on the *verifier* available at *model_dir*. 

The available verifiers are *xvector*, *vggvox*, *resnet34vox*, and *resnet50vox*. By default, xvectors are extracted 
from *300x24*-sized filterbanks, while the other vectors are extracted from *512x300x1*-sized spectrograms. 

Voice detection (*vad=[True|False]*) and augmentation (*aug=[0:no|1:aug_any|2:aug_seq|3:aug_prob]*) can be performed. 

A sample evaluation command is provided below: 

``` 
$ python ./code/test_mv.py
  --verifier "xvector"
  --data_source "/beegfs/mm10572/voxceleb2"
  --noises_dir "./data/noise"
  --model_dir "./models/xvector/model"
  --result_file "./data/vox2_imp/result_imp_xvector.csv"
  --meta_file "./data/vox2_meta/meta_vox2.csv"
  --test_paths "./data/vox2_mv/test_vox2_abspaths_1000_users"
  --test_labels "./data/vox2_mv/test_vox2_labels_1000_users"
  --test_embs "./data/vox2_mv/test_vox2_embs_1000_users.npy"
  --mv_set "./sets/std_gan_male_mv"
  --mv_type "master"
  --thr_eer 200
  --thr_far1 2000
  --enrol_size 10
  --trials 1
  --utterance_per_person 100
  --sample_rate 16000
  --preemphasis 0.97
  --frame_stride 0.01
  --frame_size 0.025
  --num_fft 512
  --min_chunk_size 10
  --max_chunk_size 300
  --aug 0
  --vad False
  --prefilter True
  --normalize True
  --nfilt 24
``` 

#### Step 4.4: Use Pre-Computed Master Voice Sets

Please find the resulting pre-trained models in the table below.
 
| Generator | Seed | Train | Master Voice Set | Any10 Impersonation Rate  | Avg10 Impersonation Rate  |
|:----:|:----------------:|:----------------:|:----------------:|:----------------:|:----------------:|
|  Spectrogram    | M | M |     [Link]()           |         ||
|  Spectrogram  |  F | F |   [Link]()              |          ||    
|  WaveGAN   | M+F  | M |    [Link]()             |       ||
|  WaveGAN   | M+F  | F |    [Link]()             |       ||
|  WaveGAN   | M  | M |    [Link]()             |       ||
|  WaveGAN   | M  | F |    [Link]()             |       ||
|  WaveGAN   | F  | M |    [Link]()             |       ||
|  WaveGAN   | F  | F |    [Link]()             |  ||

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


