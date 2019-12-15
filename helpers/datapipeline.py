#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import random
import time
import sys

from helpers.audio import decode_audio

def data_pipeline_generator(x, y, classes, augment=1, sample_rate=16000, n_seconds=3):
    """
    Function to simulate a (signal, impulse_flags), label generator
    :param x:           List of audio paths
    :param y:           List of users' labels
    :param classes:     Number of target classes
    :param augment:     Augmentation flag - 0 for non-augmentation, 1 for augmentation
    :param sample_rate: Sample rate of the audio files to be processed
    :param n_seconds:   Max number of seconds of an audio file to be processed
    :return:            (signal, impulse_flags), label
    """

    indexes = list(range(len(x)))
    random.shuffle(indexes)

    for index in indexes:

        audio = decode_audio(x[index], tgt_sample_rate=sample_rate)

        start_sample = random.choice(range(len(audio) - sample_rate*n_seconds))
        end_sample = start_sample + sample_rate*n_seconds
        audio = audio[start_sample:end_sample].reshape((1, -1, 1))

        label = y[index]

        impulse = np.random.randint(2, size=3) if augment > 0 else np.zeros(3)

        yield {'input_1': audio, 'input_2': impulse}, tf.keras.utils.to_categorical(label, num_classes=classes, dtype='float32')

    raise StopIteration()

def data_pipeline_verifier(x, y, classes, augment=1, sample_rate=16000, n_seconds=3, batch=64, prefetch=1024):
    """
    Function to create a tensorflow data pipeline for speaker verification training
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

    dataset = tf.data.Dataset.from_generator(lambda: data_pipeline_generator(x, y, classes, augment=augment, sample_rate=sample_rate, n_seconds=n_seconds),
                                             output_types=({'input_1': tf.float32, 'input_2': tf.float32}, tf.float32),
                                             output_shapes=({'input_1': [None,48000,1], 'input_2': [3]},[classes]))

    dataset = dataset.map(lambda x, y: ({'input_1': tf.squeeze(x['input_1'], axis=0), 'input_2': x['input_2']}, y))
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(prefetch)

    return dataset

def data_pipeline_gan(fps, batch, slice_len, decode_fs, decode_parallel_calls=1, slice_randomize_offset=False, slice_first_only=False, slice_overlap_ratio=0, slice_pad_end=False, repeat=False, shuffle=False, shuffle_buffer_size=None, prefetch_size=None, prefetch_gpu_num=None):
    """
    Function to create a tensorflow data pipeline for WaveGAN training
    :param fps:
    :param batch:
    :param slice_len:
    :param decode_fs:
    :param decode_parallel_calls:
    :param slice_randomize_offset:
    :param slice_first_only:
    :param slice_overlap_ratio:
    :param slice_pad_end:
    :param repeat:
    :param shuffle:
    :param shuffle_buffer_size:
    :param prefetch_size:
    :param prefetch_gpu_num:
    :return:
    """

    dataset = tf.data.Dataset.from_tensor_slices(fps)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)

    if repeat:
        dataset = dataset.repeat()

    def _decode_audio_reshaped(fp, sample_rate):
        _wav = decode_audio(fp, tgt_sample_rate=sample_rate).astype(np.float32)
        _wav = np.reshape(_wav, [_wav.shape[0], 1, 1])
        return _wav

    def _decode_audio_shaped(fp):
        _decode_audio_closure = lambda _fp: _decode_audio_reshaped(_fp, sample_rate=decode_fs)
        audio = tf.py_func(_decode_audio_closure, [fp], tf.float32, stateful=False)
        audio.set_shape([None, 1, 1])
        return audio

    dataset = dataset.map(_decode_audio_shaped, num_parallel_calls=decode_parallel_calls)

    def _slice(audio):
        if slice_overlap_ratio < 0:
            raise ValueError('Overlap ratio must be greater than 0')
        slice_hop = int(round(slice_len * (1. - slice_overlap_ratio)) + 1e-4)
        if slice_hop < 1:
            raise ValueError('Overlap ratio too high')
        if slice_randomize_offset:
            start = tf.random_uniform([], maxval=slice_len, dtype=tf.int32)
            audio = audio[start:]
        audio_slices = tf.contrib.signal.frame(audio, slice_len, slice_hop, pad_end=slice_pad_end, pad_value=0, axis=0)
        if slice_first_only:
            audio_slices = audio_slices[:1]
        return audio_slices

    def _slice_dataset_wrapper(audio):
        audio_slices = _slice(audio)
        return tf.data.Dataset.from_tensor_slices(audio_slices)

    dataset = dataset.flat_map(_slice_dataset_wrapper)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)

    dataset = dataset.batch(batch, drop_remainder=True)

    if prefetch_size is not None:
        dataset = dataset.prefetch(prefetch_size)
        if prefetch_gpu_num is not None and prefetch_gpu_num >= 0:
            dataset = dataset.apply(
                tf.data.experimental.prefetch_to_device(
                    '/device:GPU:{}'.format(prefetch_gpu_num)))

    iterator = dataset.make_one_shot_iterator()

    return iterator.get_next()