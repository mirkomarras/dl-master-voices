#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import random

from helpers.audio import decode_audio, get_tf_spectrum, get_tf_filterbanks, play_n_rec

def data_pipeline_generator_verifier(x, y, classes, sample_rate=16000, n_seconds=3):
    """
    Function to simulate a (signal, impulse_flags), label generator for training a verifier
    :param x:           List of audio paths
    :param y:           List of users' labels
    :param classes:     Number of target classes
    :param augment:     Augmentation flag - 0 for non-augmentation, 1 for augmentation
    :param sample_rate: Sample rate of the audio files to be processed
    :param n_seconds:   Max number of seconds of an audio file to be processed
    :return:            (signal, impulse_flags), label
    """

    while True:
        indexes = list(range(len(x)))
        random.shuffle(indexes)
        for index in indexes:
            audio = decode_audio(x[index], tgt_sample_rate=sample_rate)
            start_sample = random.choice(range(len(audio) - sample_rate*n_seconds))
            end_sample = start_sample + sample_rate*n_seconds
            audio = audio[start_sample:end_sample].reshape((1, -1, 1))
            label = y[index]
            impulse = np.random.randint(2, size=3)
            yield {'input_1': audio, 'input_2': impulse}, tf.keras.utils.to_categorical(label, num_classes=classes, dtype='float32')

    raise StopIteration()

def data_pipeline_verifier(x, y, classes, sample_rate=16000, n_seconds=3, batch=64, prefetch=1024):
    """
    Function to create a tensorflow data pipeline for training a verifier
    :param x:           List of audio paths
    :param y:           List of users' labels
    :param classes:     Number of target classes
    :param augment:     Augmentation flag - 0 for non-augmentation, 1 for augmentation
    :param sample_rate: Sample rate of the audio files to be processed
    :param n_seconds:   Max number of seconds of an audio file to be processed
    :param batch:       Size of a training batch
    :param prefetch:    Number of prefetched batches
    :return:            Data pipeline
    """

    dataset = tf.data.Dataset.from_generator(lambda: data_pipeline_generator_verifier(x, y, classes, sample_rate=sample_rate, n_seconds=n_seconds), output_types=({'input_1': tf.float32, 'input_2': tf.float32}, tf.float32), output_shapes=({'input_1': [None, sample_rate*n_seconds, 1], 'input_2': [3]}, [classes]))
    dataset = dataset.map(lambda x, y: ((tf.squeeze(x['input_1'], axis=0), x['input_2']), y))
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(prefetch)
    return dataset

def data_pipeline_generator_gan(x, slice_len, sample_rate=16000):
    """
    Function to simulate a signal generator for training a gan
    :param x:           List of audio paths
    :param slice_len:   Length of each audio sample
    :param sample_rate: Sample rate of the audio files to be processed
    :return:            (signal)
    """
    indexes = list(range(len(x)))
    random.shuffle(indexes)

    for index in indexes:
        audio = decode_audio(x[index], tgt_sample_rate=sample_rate)
        start_sample = random.choice(range(len(audio) - slice_len))
        end_sample = start_sample + slice_len
        audio = audio[start_sample:end_sample].reshape((1, -1, 1))
        yield audio

    raise StopIteration()

def data_pipeline_gan(x, slice_len, sample_rate=16000, batch=64, prefetch=1024, output_type='raw'):
    """
    Function to create a tensorflow data pipeline for training a gan
    :param x:           List of audio paths
    :param slice_len:   Length of each audio sample
    :param sample_rate: Sample rate of the audio files to be processed
    :param batch:       Size of a training batch
    :param prefetch:    Number of prefetched batches
    :return:            Data pipeline
    """

    dataset = tf.data.Dataset.from_generator(lambda: data_pipeline_generator_gan(x, slice_len=slice_len, sample_rate=sample_rate),
                                             output_types=(tf.float32),
                                             output_shapes=([None, slice_len, 1]))

    dataset = dataset.map(lambda x: tf.squeeze(x, axis=0))
    dataset = dataset.batch(batch)

    if output_type == 'spectrum':
        dataset = dataset.map(lambda x: tf.pad(x, [[0, 0], [0, 128], [0, 0]], 'CONSTANT'))
        dataset = dataset.map(lambda x: get_tf_spectrum(x, frame_size=0.016, frame_stride=0.008, num_fft=256))

    dataset = dataset.prefetch(prefetch)

    return dataset

def data_pipeline_generator_mv(x, sample_rate=16000, n_seconds=3):
    """
    Function to simulate a signal generator for training a master voice vocoder
    :param x:           List of audio paths
    :param sample_rate: Sample rate of the audio files to be processed
    :param n_seconds:   Max number of seconds of an audio file to be processed
    :return:            (Signal)
    """
    indexes = list(range(len(x)))
    random.shuffle(indexes)

    for index in indexes:
        audio = decode_audio(x[index], tgt_sample_rate=sample_rate)
        start_sample = random.choice(range(len(audio) - sample_rate*n_seconds))
        end_sample = start_sample + sample_rate*n_seconds
        audio = audio[start_sample:end_sample].reshape((1, -1, 1))
        yield audio

    raise StopIteration()

def data_pipeline_mv(x, sample_rate=16000, n_seconds=3, batch=64, prefetch=1024):
    """
    Function to create a tensorflow data pipeline for training a master voice vocoder
    :param x:           List of audio paths
    :param sample_rate: Sample rate of the audio files to be processed
    :param n_seconds:   Max number of seconds of an audio file to be processed
    :param batch:       Size of a training batch
    :param prefetch:    Number of prefetched batches
    :return:            Data pipeline
    """

    dataset = tf.data.Dataset.from_generator(lambda: data_pipeline_generator_mv(x, sample_rate=sample_rate, n_seconds=n_seconds),
                                             output_types=(tf.float32),
                                             output_shapes=([None, sample_rate*n_seconds, 1]))

    dataset = dataset.map(lambda x: tf.squeeze(x, axis=0))
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(prefetch)

    return dataset