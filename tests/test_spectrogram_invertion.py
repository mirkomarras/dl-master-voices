import soundfile as sf
import numpy as np

from helpers import plotting

from helpers.audio import decode_audio, get_np_spectrum, denormalize_frames, spectrum_to_signal

def sample_wave():
    audio_path = './tests/original_audio.wav'
    xt = decode_audio(audio_path).astype(np.float32)
    return xt


def compare_inverted_spectrum(original_audio, full=False, sample_rate=16000):
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

    #GRIFFIN-LIM IS RUN HERE
    inv_signal = spectrum_to_signal(sp.T, len(original_audio))

    #EXPORTING
    fig = plotting.images(sp, cmap='hsv')
    sf.write('./tests/inverted_' + ('full_' if full else 'half_') + 'audio.wav', inv_signal, sample_rate)
    fig.savefig('./tests/inverted_' + ('full_' if full else 'half_') + 'audio.png')


    #### UNIT TESTS ####
    gla_xt = decode_audio('./tests/inverted_' + ('full_' if full else 'half_') + 'audio.wav').astype(np.float32)

    #test for consistency with the final wav files
    assert gla_xt.shape == original_audio.shape               # wav output have same shape
    assert (gla_xt == original_audio).all() == False          # not exactly same audio (error exists in reconstruction)
    assert np.mean(np.abs(gla_xt-original_audio)) < 0.1       # within 10% margin of difference for all elements
    



# for use the pytest
def test_cis():
    compare_inverted_spectrum(sample_wave(), full=True)
    compare_inverted_spectrum(sample_wave(), full=False)


#normal python
if __name__ == '__main__':
    compare_inverted_spectrum(sample_wave(), full=True)
    compare_inverted_spectrum(sample_wave(), full=False)
