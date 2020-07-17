
import os
import numpy as np

import pytest

from models import gan



def test_pretrained_msgan_voxceleb(height=256, aspect=1, version=0):
    dataset = 'voxceleb'
    samples = 32

    gan_ = gan.MultiscaleGAN(dataset, version=version, patch=height, width_ratio=aspect, min_output=8)
    assert gan_.dirname() == f'./data/models/gan/ms-gan/voxceleb/v{version:03d}'
    assert os.path.isdir(gan_.dirname()), f'The pretrained models seems to be missing {gan_.dirname()}'

    gan_.load()

    # Sample from the model
    batch_y = gan_.sample(samples).numpy()
    assert batch_y.shape == (samples, 256, 256, 1)
    
    # Test sample values - very loose ends, but should be enough to detect untrained models
    assert batch_y.min() < -1, f'The minimum intensity in the sample seems to be invalid {batch_y.min()}'
    assert batch_y.max() > 1, f'The maximum intensity in the sample seems to be invalid {batch_y.min()}'
    