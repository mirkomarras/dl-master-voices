import pytest
import numpy as np

from helpers.audio import decode_audio, get_tf_spectrum, get_np_spectrum


def rolled_norm(a, b, shift=4):
    l2 = np.inf
    best_shift = None
    for rx in range(-shift, shift):
        for ry in range(-shift, shift):
            c = np.roll(b, rx, axis=0)
            c = np.roll(c, ry, axis=1)
            L2 = np.linalg.norm(a - c)
            if L2 < l2:
                l2 = L2
                best_shift = (rx, ry)
    return l2, best_shift


@pytest.fixture(scope="session")
def sample_wave():
    audio_path = './tests/original_audio.wav'
    xt = decode_audio(audio_path).astype(np.float32)
    return xt


def test_compare_np_tf(sample_wave):
    
    assert len(sample_wave) == 76161

    # Numpy spectrum
    sp_np, _, _ = get_np_spectrum(sample_wave, 16000, num_fft=512)
    print('> numpy spectrum:', sp_np.shape, np.min(sp_np), np.max(sp_np))

    # TF Spectrum
    sp_tf = np.squeeze(get_tf_spectrum(sample_wave.reshape((1, -1, 1)), num_fft=512).numpy())
    print('> tensorflow spectrum:', sp_tf.shape, np.min(sp_tf), np.max(sp_tf))
    
    # Validate shapes
    assert sp_np.shape == (256, 474)
    assert sp_tf.shape == (256, 474)

    # Check data ranges - are the min / max values withing 5% tolerance
    min_sp = np.min(sp_np)
    max_sp = np.max(sp_np)
    min_tf = np.min(sp_tf)
    max_tf = np.max(sp_tf)
    assert np.abs((max_sp - max_tf) / max_sp) < 0.05
    assert np.abs((min_sp - min_tf) / min_sp) < 0.05

    # Check spectrogram similarity
    L2_norm, best_shift = rolled_norm(sp_np, sp_tf)
    
    assert L2_norm < 25
    assert best_shift == (0, 0)
