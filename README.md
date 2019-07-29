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

### Step 2: Train and Evaluate a Speaker Verifier

The training script looks for utterances in *data_source_vox1* and *data_source_vox2* folders, excluding the utterances 
from users involved in master voice analysis. Both data sources must point to a folder including *dev* and *test* subfolders. 

The available verifiers are *xvector*, *vggvox*, *resnet34vox*, and *resnet50vox*. By default, xvector models
are trained on *300x24*-sized filterbanks and return vectors of size *1024*, while the other models are trained on 
*512x300x1*-sized spectrograms and return vectors of size *512*. Each model is trained for *n_epochs=40* on *batch_size=32* with *learning_rate=1e-1*.
The script saves the pre-trained model into the folder *model_dir*.

Voice detection (*vad=[True|False]*) and augmentation (*aug=[0:no|1:aug_any|2:aug_seq|3:aug_prob]*) can be performed. 

A sample training command is provided below: 

```
$ python ./train/train_speaker_verifier.py 
  --verifier "xvector"
  --data_source_vox1 "/beegfs/mm10572/voxceleb1" 
  --data_source_vox2 "/beegfs/mm10572/voxceleb2" 
  --noises_dir "./data/noise"
  --model_dir "./models/xvector/model"
  --aug 3 
  --vad True 
```

Please find the resulting pre-trained models in the table below.
 
| Name | Pre-Trained Model | Equal Error Rate  |
|:----:|:----------------:|:----------------:|
|  X-Vector    |       [Link]()           |        |        
|  VGGVox-Vector    |      [Link]()              |    |            
|  ResNet34Vox-Vector    |       [Link]()             |   |             
|  ResNet50Vox-Vector    |      [Link]()              |    |            

### Step 3: Train a Generative Adversarial Network (GAN)

Please find the resulting pre-trained models in the table below.
 
| Name | Seed | Pre-Trained Model | 
|:----:|:----------------:|:----------------:|
|  WaveGAN    | M+F |      [Link]()           |         
|  WaveGAN   |   M |   [Link]()              |              
|  WaveGAN    |  F |     [Link]()             |        

### Step 4: Generate and Evaluate a Master Voice

Please find the resulting pre-trained models in the table below.
 
| Generator | Seed | Train | Master Voice Set | Any10 Impersonation Rate  | AVG10 Impersonation Rate  |
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


