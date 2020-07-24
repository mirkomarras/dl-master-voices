import pytest
import numpy as np
from helpers.audio import decode_audio


'''
All session fixtures should go here
'''

@pytest.fixture(scope="session")
def sample_wave():
    audio_path = './tests/original_audio.wav'
    xt = decode_audio(audio_path).astype(np.float32)
    return xt