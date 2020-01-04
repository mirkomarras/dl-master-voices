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
Please go a to ```/beegfs/{id}``` directory, then:

``` 
git clone --single-branch --branch mv_fwk https://github.com/mirkomarras/dl-master-voices.git
cd dl-master-voices
chmod +x ./install.sh
./install.sh
``` 

This creates a virtual env in ```/beegfs/{id}/dl-master_voices/mvenv``` folder, with all the needed Python packages.
It also downloads project data in ```/beegfs/{id}/dl-master-voices/data``` folder. 

## Getting Started

### Train a speaker verification network
To start training a sample speaker verification model, from the project folder, please run:

``` 
sbatch ./sbatch/train_verifier.sbatch
``` 

The output of the job is saved at ``` ./jobs/slurm_train_verifier_{job_id}.out ```.

The model weights are saved at ```./data/pt_models/{xvector|vggvox|resnet34vox|resnet50vox}/v{version_id}/model_weights.tf```.  

### Test a speaker verification network
To start testing a sample speaker verification model, from the project folder, please run:

``` 
sbatch ./sbatch/test_verifier.sbatch
``` 

The output of the job is saved at ``` ./jobs/slurm_test_verifier_{job_id}.out ```.

### Train a generative adversarial network
To start training a sample generative adversarial model, from the project folder, please run:

``` 
sbatch ./sbatch/train_gan.sbatch
``` 

The output of the job is saved at ``` ./jobs/slurm_train_wavegan_{job_id}.out ```.

The generator model is saved at ```./data/pt_models/wavegan/{neutral|male|female}/v{version_id}/generator_weights.tf```.

The discriminator model is saved at ```./data/pt_models/wavegan/{neutral|male|female}/v{version_id}/discriminator_weights.tf```.  

### Test a generative adversarial network
To start testing a sample generative adversarial model, from the project folder, please run:

``` 
sbatch ./sbatch/test_gan.sbatch
``` 

The output of the job is saved at ``` ./jobs/slurm_test_wavegan_{job_id}.out ```.

Preview samples are saved at ```./data/pt_models/wavegan/{neutral|male|female}/v{version_id}/fake.wav ```.  

### Optimize a master voice
To start optimizing a master voice, from the project folder, please run:

``` 
sbatch ./sbatch/train_mv.sbatch
``` 

The output of the job is saved at ``` ./jobs/slurm_train_mv_{job_id}.out ```.

The master voice sample are saved at ```./data/vs_mv_data/{net}-{netv}_{gan}-{ganv}_{f|m}-{f|m}_{mv|sv}/v{version_id}```.  

### Test a master voice
To start testing a master voice population, from the project folder, please run:

``` 
sbatch ./sbatch/test_mv.sbatch
``` 

The output of the job is saved at ``` ./jobs/slurm_test_mv_{job_id}.out ```.

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


