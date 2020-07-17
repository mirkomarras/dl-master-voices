
import os

import pytest

import helpers.datapipeline as dp


@pytest.fixture(scope="session")
def digits(examples=128):
    audio_dir = './data/digits/train'
    assert os.path.isdir(audio_dir), f'Data directory not found! {audio_dir}'    

    # Find files and check availability
    x_train = [os.path.join(audio_dir, x) for x in os.listdir(audio_dir)]
    x_train = x_train[:examples]
    assert len(x_train) == examples, f'Invalid number of test samples - expected {examples} got {len(x_train)}'

    return x_train


def test_vanilla_pipeline(digits, sample_rate=16000, batch=32, length=1):
    # Basic setup
    length *= sample_rate

    # Vanilla data pipeline
    train_data = dp.data_pipeline_gan(digits, slice_len=length, sample_rate=sample_rate, batch=batch, output_type='spectrum') # pad_width='auto', resize=resize
    assert train_data.element_spec.shape[1:] == (256, 98, 1), f'Invalid shape of samples in the *vanilla* data pipeline: {train_data.element_spec.shape}'
    
    counter = 0
    for index, x in enumerate(train_data):
        assert x.shape == (batch, 256, 98, 1)
        print(f'> batch {index} {x.shape}')
        counter += 1
    
    assert counter == 4, 'Invalid number of batches'


def test_resized_pipeline(digits, sample_rate=16000, batch=32, length=1):
    # Basic setup
    length *= sample_rate

    # Resized data pipeline - elements are resized to 128px height (with preserved aspect ratio)
    train_data = dp.data_pipeline_gan(digits, slice_len=length, sample_rate=sample_rate, batch=batch, output_type='spectrum', resize=128)
    assert train_data.element_spec.shape[1:] == (128, 49, 1), f'Invalid shape of samples in the *resized* data pipeline: {train_data.element_spec.shape}'
    
    counter = 0
    for index, x in enumerate(train_data):
        assert x.shape == (batch, 128, 49, 1)
        print(f'> batch {index} {x.shape}')
        counter += 1
    
    assert counter == 4, 'Invalid number of batches'


def test_padded_pipeline(digits, sample_rate=16000, batch=32, length=1):
    # Basic setup
    length *= sample_rate

    # Padded data pipeline - pad elements to correct width (2^i)
    train_data = dp.data_pipeline_gan(digits, slice_len=length, sample_rate=sample_rate, batch=batch, output_type='spectrum', pad_width='auto')
    assert train_data.element_spec.shape[1:] == (256, 128, 1), f'Invalid shape of samples in the *padded* data pipeline: {train_data.element_spec.shape}'
    
    counter = 0
    for index, x in enumerate(train_data):
        assert x.shape == (batch, 256, 128, 1)
        print(f'> batch {index} {x.shape}')
        counter += 1
    
    assert counter == 4, 'Invalid number of batches'

