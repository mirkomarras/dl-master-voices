import soundfile as sf
import numpy as np

from helpers.audio import decode_audio, get_np_spectrum, denormalize_frames, spectrum_to_signal

def sample_wave():
    audio_path = './tests/original_audio.wav'
    xt = decode_audio(audio_path).astype(np.float32)
    return xt

def test_compare_inverted_spectrum(original_audio, symmetric=False, sample_rate=16000):
    print('> numpy waveform:', original_audio.shape, np.min(original_audio), np.max(original_audio))

    sp_norm, sp_avg, sp_std = get_np_spectrum(original_audio, sample_rate, num_fft=512, symmetric=symmetric)
    print('> numpy spectrum:', sp_norm.shape, np.min(sp_norm), np.max(sp_norm))

    sp = denormalize_frames(sp_norm, sp_avg, sp_std)
    print('> numpy denormalized spectrum:', sp.shape, np.min(sp), np.max(sp))

    if not symmetric:
        sp = np.vstack((sp, sp[::-1]))
        sp = np.hstack((sp, np.tile(sp[:, [-1]], 1)))
        print('> numpy mirrored spectrum:', sp.shape, np.min(sp), np.max(sp))

    inv_signal = spectrum_to_signal(sp.T, len(original_audio))
    sf.write('./tests/inverted_audio.wav', inv_signal, sample_rate)

if __name__ == '__main__':
    test_compare_inverted_spectrum(sample_wave())
