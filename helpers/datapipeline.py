import tensorflow as tf

def mfcc(pcm, sample_rate=16000):
    # https://www.tensorflow.org/api_docs/python/tf/signal/mfccs_from_log_mel_spectrograms
    # A Tensor of [batch_size, num_samples] mono PCM samples in the range [-1, 1].
    
    # A 1024-point STFT with frames of 64 ms and 75% overlap.
    stfts = tf.contrib.signal.stft(pcm, frame_length=1024, frame_step=256, fft_length=512)
    spectrograms = tf.abs(stfts)
    print(pcm)
    print(pcm.shape)
    print(stfts.shape)

    # Warp the linear scale spectrograms into the mel-scale.
    num_spectrogram_bins = 512 # stfts.shape[-1].value
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 80
    linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix( num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz, upper_edge_hertz)
    mel_spectrograms = tf.tensordot( spectrograms, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate( linear_to_mel_weight_matrix.shape[-1:]))

    # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)

    # Compute MFCCs from log_mel_spectrograms and take the first 13.
    mfccs = tf.contrib.signal.mfccs_from_log_mel_spectrograms( log_mel_spectrograms)[..., :13]
    return mfccs

def spectrogram(audio, label, sample_rate=16000, frame_size=0.025, frame_stride=0.01, num_fft=512):

    audio = tf.reshape(audio, (-1, ))
    # print(audio)
    # print(audio.shape)
    frames = tf.contrib.signal.frame(audio, frame_length=int(frame_size * sample_rate), frame_step=int(frame_stride * sample_rate), pad_end=True, name="Frames")
    frames = frames * tf.contrib.signal.hamming_window(int(frame_size * sample_rate), periodic=True)
    frames = tf.transpose(frames)
    t = tf.shape(frames)
    pad_amount = tf.zeros([int(num_fft) - int(frame_size * sample_rate), t[1]], tf.float32)
    frames_pad = tf.concat([frames, pad_amount], axis=0)
    
    # Computing the FFT of the audio tensor
    y = tf.cast(frames_pad, tf.complex64)
    y = tf.transpose(y)
    spec = tf.cast(tf.abs(tf.spectral.fft(y, name="FFT")), tf.float32)
    mag_spec = tf.transpose(spec)
    
    # Normalizing the spectrogram
    mean_tensor, variance_tensor = tf.nn.moments(mag_spec, axes=[1])
    std_tensor = tf.math.sqrt(variance_tensor)
    m_shape = tf.shape(mean_tensor)
    s_shape = tf.shape(std_tensor)
    spec_norm = (mag_spec - tf.reshape(mean_tensor, [m_shape[0], 1])) / tf.maximum(tf.reshape(std_tensor, [s_shape[0], 1]), 1e-8)
    spec_norm = tf.expand_dims(spec_norm, 0)
    spec_norm = tf.expand_dims(spec_norm, 3)

    return spec_norm, label


def playback_n_recording(speech, label, speaker, room, mic, max_length=48000):

    speech = speech[:, :max_length, :]
    
    speaker_out = tf.nn.conv1d(speech, speaker, 1, padding="SAME")
    noise_tensor = tf.random.normal(tf.shape(speech), mean=0, stddev=5e-3, dtype=tf.float32)
    speaker_out = tf.add(speaker_out, noise_tensor)
    #room_out = tf.nn.conv1d(speaker_out, room, 1, padding="SAME")
    #audio_out = tf.nn.conv1d(room_out, mic, 1, padding="SAME")
    
    return speaker_out, label

def speech_only(speech, label, speaker, room, mic, max_length=48000):
    return speech[:, :max_length, :], label

def clip(speech, label, speaker, room, mic, max_length=48000):
    return speech[:, :max_length, :], label, speaker, room, mic
