import soundfile as sf
import numpy as np

from helpers import plotting

from helpers.audio import decode_audio, get_np_spectrum, denormalize_frames, spectrum_to_signal

def sample_wave():
    audio_path = './tests/original_audio.wav'
    xt = decode_audio(audio_path).astype(np.float32)
    return xt

def test_compare_inverted_spectrum(original_audio, full=False, sample_rate=16000):
    print('> numpy waveform:', original_audio.shape, np.min(original_audio), np.max(original_audio))

    sp_norm, sp_avg, sp_std = get_np_spectrum(original_audio, sample_rate, num_fft=512, full=full)
    print('> numpy spectrogram:', sp_norm.shape, np.min(sp_norm), np.max(sp_norm))
    print('> numpy avg:', sp_avg.shape, np.min(sp_avg), np.max(sp_avg))
    print('> numpy std:', sp_std.shape, np.min(sp_std), np.max(sp_std))

    sp = denormalize_frames(sp_norm, sp_avg, sp_std)
    print('> numpy denormalized spectrogram:', sp.shape, np.min(sp), np.max(sp))

    if not full:
        sp = np.vstack((sp, np.zeros((1, sp.shape[1])), sp[:0:-1]))
        print('> numpy mirrored spectrogram:', sp.shape, np.min(sp), np.max(sp))

    inv_signal = spectrum_to_signal(sp.T, len(original_audio))
    fig = plotting.imsc(sp, cmap='hsv')
    sf.write('./tests/inverted_' + ('full_' if full else 'half_') + 'audio.wav', inv_signal, sample_rate)
    fig.savefig('./tests/inverted_' + ('full_' if full else 'half_') + 'audio.png')
    return sp


if __name__ == '__main__':
    test_compare_inverted_spectrum(sample_wave(), full=True)
    test_compare_inverted_spectrum(sample_wave(), full=False)
