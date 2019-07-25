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

...

## Getting Started

### Step 1: Download Source Data Sets
[VoxCeleb1](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html) contains over 100,000 utterances for 1,251 
celebrities, extracted from videos uploaded to YouTube. 

[VoxCeleb2](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html) contains over 1 million utterances for 6,112 
celebrities, extracted from videos uploaded to YouTube.

### Step 2: Train Speaker Verification Algorithms

The training script looks for utterances in *data_source_vox1* and *data_source_vox2* folders, excluding the ones from users 
used for master voice analysis. Both data sources should point to a folder containing *dev* and *test* VoxCeleb folders. The
available verifiers are the following ones: xvector, vggvox, resnet34vox, and resnet50vox. By 
default, the model is trained for *n_epochs=40* on *512x300*-sized spectrograms grouped in batches of size *batch_size=32* with a learning rate of 
*learning_rate=1e-1*. For each utterance, a resnet50-vector of size *512* is returned. On each utterance, voice activity detection (*vad=[True|False]*) and data augmentation (*aug=[0:no|1:aug_any|2:aug_seq|3:aug_prob]*) can be performed. This step saves a x-vector model on the folder 
*model_dir*.

A sample training command is provided below: 

```
$ python ./train/train_speaker_verifier.py 
  --verifier "xvector"
  --data_source_vox1 "/beegfs/mm10572/voxceleb1" 
  --data_source_vox2 "/beegfs/mm10572/voxceleb2" 
  --noises_dir "./data/noise"
  --model_dir "./models/pre-trained-resnet50vox"
  --aug 3 
  --vad True 
```

Please find the resulting pre-trained models trained on 1,024,292 utterances from 5,205 users in *./models/*. 

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


