from scipy.signal import lfilter, butter
import tensorflow as tf
import soundfile as sf
import numpy as np
import librosa
import queue
import time
import decimal
import math
import logging
import os
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def __conv_bn_pool(inp_tensor, layer_idx, conv_filters, conv_kernel_size, conv_strides, conv_pad, pool='', pool_size=(2, 2), pool_strides=None, conv_layer_prefix='conv'):
    x = tf.keras.layers.ZeroPadding2D(padding=conv_pad, name='pad{}'.format(layer_idx))(inp_tensor)
    x = tf.keras.layers.Conv2D(filters=conv_filters, kernel_size=conv_kernel_size, strides=conv_strides, padding='valid', name='{}{}'.format(conv_layer_prefix, layer_idx))(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=1., name='bn{}'.format(layer_idx))(x)
    x = tf.keras.layers.Activation('relu', name='relu{}'.format(layer_idx))(x)
    if pool == 'max':
        x = tf.keras.layers.MaxPooling2D(pool_size=pool_size, strides=pool_strides, name='mpool{}'.format(layer_idx))(x)
    elif pool == 'avg':
        x = tf.keras.layers.AveragePooling2D(pool_size=pool_size, strides=pool_strides, name='apool{}'.format(layer_idx))(x)
    return x

def __conv_bn_dynamic_apool(inp_tensor, layer_idx, conv_filters, conv_kernel_size, conv_strides, conv_pad, conv_layer_prefix='conv'):
    x = tf.keras.layers.ZeroPadding2D(padding=conv_pad, name='pad{}'.format(layer_idx))(inp_tensor)
    x = tf.keras.layers.Conv2D(filters=conv_filters, kernel_size=conv_kernel_size, strides=conv_strides, padding='valid', name='{}{}'.format(conv_layer_prefix, layer_idx))(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=1., name='bn{}'.format(layer_idx))(x)
    x = tf.keras.layers.Activation('relu', name='relu{}'.format(layer_idx))(x)
    x = tf.keras.layers.AveragePooling2D(pool_size=(1, 8), strides=(1, 1), name='apool{}'.format(layer_idx))(x)
    x = tf.math.reduce_mean(x, axis=[1, 2], name='rmean{}'.format(layer_idx))
    x = tf.math.l2_normalize(x)
    return x

def round_half_up(number):
    return int(decimal.Decimal(number).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))

def rolling_window(a, window, step=1):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)[::step]

def framesig(sig, frame_len, frame_step, winfunc=lambda x: np.ones((x,)), stride_trick=True):
    slen = len(sig)
    frame_len = int(round_half_up(frame_len))
    frame_step = int(round_half_up(frame_step))
    if slen <= frame_len:
        numframes = 1
    else:
        numframes = 1 + int(math.ceil((1.0 * slen - frame_len) / frame_step)) # LV

    padlen = int((numframes - 1) * frame_step + frame_len)

    zeros = np.zeros((padlen - slen,))
    padsignal = np.concatenate((sig, zeros))
    if stride_trick:
        win = winfunc(frame_len)
        frames = rolling_window(padsignal, window=frame_len, step=frame_step)
    else:
        indices = np.tile(np.arange(0, frame_len), (numframes, 1)) + np.tile(
            np.arange(0, numframes * frame_step, frame_step), (frame_len, 1)).T
        indices = np.array(indices, dtype=np.int32)
        frames = padsignal[indices]
        win = np.tile(winfunc(frame_len), (numframes, 1))

    return frames * win

def magspec(frames, NFFT):
    if np.shape(frames)[1] > NFFT:
        logging.warn(
            'frame length (%d) is greater than FFT size (%d), frame will be truncated. Increase NFFT to avoid.',
            np.shape(frames)[1], NFFT)
    complex_spec = np.fft.rfft(frames, NFFT)
    return np.absolute(complex_spec)

def powspec(frames, NFFT):
    return 1.0 / NFFT * np.square(magspec(frames, NFFT))

def logpowspec(frames, NFFT, norm=1):
    ps = powspec(frames, NFFT)
    ps[ps <= 1e-30] = 1e-30
    lps = 10 * np.log10(ps)
    if norm:
        return lps - np.max(lps)
    else:
        return lps

def preemphasis(signal, coeff=0.95):
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])

def build_buckets(max_sec, step_sec, frame_step):
    buckets = {}
    frames_per_sec = int(1 / frame_step)
    end_frame = int(max_sec * frames_per_sec)
    step_frame = int(step_sec * frames_per_sec)
    for i in range(0, end_frame + 1, step_frame):
        s = i
        s = np.floor((s - 7 + 2) / 2) + 1  # conv1
        s = np.floor((s - 3) / 2) + 1  # mpool1
        s = np.floor((s - 5 + 2) / 2) + 1  # conv2
        s = np.floor((s - 3) / 2) + 1  # mpool2
        s = np.floor((s - 3 + 2) / 1) + 1  # conv3
        s = np.floor((s - 3 + 2) / 1) + 1  # conv4
        s = np.floor((s - 3 + 2) / 1) + 1  # conv5
        s = np.floor((s - 3) / 2) + 1  # mpool5
        s = np.floor((s - 1) / 1) + 1  # fc6
        if s > 0:
            buckets[i] = int(s)
    return buckets

def load_wav(filename, sample_rate):
    audio, sr = librosa.load(filename, sr=sample_rate, mono=True)
    audio = audio.flatten()
    return audio

def normalize_frames(m,epsilon=1e-12):
    return np.array([(v - np.mean(v)) / max(np.std(v),epsilon) for v in m])

def remove_dc_and_dither(sin, sample_rate):
    if sample_rate == 16e3:
        alpha = 0.99
    elif sample_rate == 8e3:
        alpha = 0.999
    else:
        print("Sample rate must be 16kHz or 8kHz only")
        exit(1)
    sin = lfilter([1,-1], [1,-alpha], sin)
    dither = np.random.random_sample(len(sin)) + np.random.random_sample(len(sin)) - 1
    spow = np.std(dither)
    sout = sin + 1e-6 * spow * dither
    return sout

def get_fft_spectrum(signal, buckets):
    signal *= 2**15

    signal = remove_dc_and_dither(signal, 16000)
    signal = preemphasis(signal, coeff=0.97)
    frames = framesig(signal, frame_len=0.025*16000, frame_step=0.01*16000, winfunc=np.hamming)
    fft = abs(np.fft.fft(frames,n=512))
    fft_norm = normalize_frames(fft.T)

    rsize = max(k for k in buckets if k <= fft_norm.shape[1])
    rstart = int((fft_norm.shape[1]-rsize)/2)
    out = fft_norm[:,rstart:rstart+rsize]

    return np.expand_dims(out, axis=2)

def data_pipeline_generator_verifier(x, y, classes, sample_rate=16000, n_seconds=4):
    buckets = build_buckets(10, 1, 0.01)
    indexes = list(range(len(x)))
    random.shuffle(indexes)

    for index in indexes:
        audio = load_wav(x[index], sample_rate)
        audio = audio[:sample_rate*n_seconds]
        out = get_fft_spectrum(audio, buckets)[:512,:300,:]
        label = y[index]
        yield out, tf.keras.utils.to_categorical(label, num_classes=classes, dtype='float32')

    raise StopIteration()

def data_pipeline_verifier(x, y, classes):
    dataset = tf.data.Dataset.from_generator(lambda: data_pipeline_generator_verifier(x, y, classes), output_types=(tf.float32, tf.float32), output_shapes=([512,300,1],[classes]))
    dataset = dataset.batch(16)
    dataset = dataset.prefetch(16)
    return dataset.repeat()

from keras.callbacks import LearningRateScheduler
class StepDecay():
    def __init__(self, init_alpha=0.01, factor=0.25, drop_every=10):
        self.init_alpha = init_alpha
        self.factor = factor
        self.drop_every = drop_every

    def __call__(self, epoch):
        exp = np.floor((1 + epoch) / self.drop_every)
        alpha = self.init_alpha * (self.factor ** exp)
        print('Learning rate for next epoch', float(alpha))
        return float(alpha)

if __name__ == '__main__':

    print('Preparing data')
    x_data = []
    y_data = []
    n_users = 1000
    sample_rate = 16000
    n_seconds = 3
    source_dir = '/beegfs/mm10572/voxceleb1/dev'

    user_count = 0
    for user_id, user_dir in enumerate(os.listdir(os.path.join(source_dir))):
        print('\rUser', user_id+1, 'of', n_users, end='')
        for video_id, video_dir in enumerate(os.listdir(os.path.join(source_dir, user_dir))):
            for audio_id, audio_file in enumerate(os.listdir(os.path.join(source_dir, user_dir, video_dir))):
                x_data.append(os.path.join(source_dir, user_dir, video_dir, audio_file))
                y_data.append(user_id)
        if user_id+1 >= n_users:
            break
    print()

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    print('Found', len(x_data), 'samples')
    print('Found', n_users, 'users')

    # Model creation
    signal_input = tf.keras.Input(shape=(512,None,1,))
    x = __conv_bn_pool(signal_input, layer_idx=1, conv_filters=96, conv_kernel_size=(7, 7), conv_strides=(2, 2), conv_pad=(1, 1), pool='max', pool_size=(3, 3), pool_strides=(2, 2))
    x = __conv_bn_pool(x, layer_idx=2, conv_filters=256, conv_kernel_size=(5, 5), conv_strides=(2, 2), conv_pad=(1, 1), pool='max', pool_size=(3, 3), pool_strides=(2, 2))
    x = __conv_bn_pool(x, layer_idx=3, conv_filters=384, conv_kernel_size=(3, 3), conv_strides=(1, 1), conv_pad=(1, 1))
    x = __conv_bn_pool(x, layer_idx=4, conv_filters=256, conv_kernel_size=(3, 3), conv_strides=(1, 1), conv_pad=(1, 1))
    x = __conv_bn_pool(x, layer_idx=5, conv_filters=256, conv_kernel_size=(3, 3), conv_strides=(1, 1), conv_pad=(1, 1), pool='max', pool_size=(5, 3), pool_strides=(3, 2))
    x = __conv_bn_dynamic_apool(x, layer_idx=6, conv_filters=4096, conv_kernel_size=(9, 1), conv_strides=(1, 1), conv_pad=(0, 0), conv_layer_prefix='fc')
    x = tf.keras.layers.Dense(1024, name='fc7')(x)
    x = tf.keras.layers.Dense(n_users, activation='softmax', name='fc8')(x)

    model = tf.keras.Model(inputs=[signal_input], outputs=[x])
    model.summary()
    schedule = StepDecay(init_alpha=1e-2, factor=0.1, drop_every=10)
    callbacks = [LearningRateScheduler(schedule)]
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-2), loss='categorical_crossentropy', metrics=['accuracy'])

    # Model training
    train_data = data_pipeline_verifier(x_data, y_data, n_users)
    model.fit(train_data, steps_per_epoch=len(x_data)//32, epochs=10, callbacks=callbacks)