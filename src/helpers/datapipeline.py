#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import random
import time
import sys

from src.helpers.audio import decode_audio

def data_pipeline_generator(x, y, sample_rate=16000, n_seconds=3, chunk_size=100000):

    for i in range(0, len(x), chunk_size):
        x_chunk = x[i:i+chunk_size]
        y_chunk = y[i:i+chunk_size]
        for index in range(len(x_chunk)):
            audio = decode_audio(x_chunk[index], sample_rate=sample_rate)
            start_sample_id = random.choice(range(len(audio) - sample_rate*n_seconds))
            end_sample_id = start_sample_id + sample_rate*n_seconds
            assert end_sample_id > 0 and end_sample_id < len(audio)
            audio = audio.reshape((1, -1, 1))[:, start_sample_id:end_sample_id, :]
            label = y_chunk[index]
            yield audio, label

    raise StopIteration()

def data_pipeline_verifier(x, y, sample_rate=16000, n_seconds=3, buffer_size=25000, batch=64, prefetch=1024):
    dataset = tf.data.Dataset.from_generator(lambda: data_pipeline_generator(x, y, sample_rate=sample_rate, n_seconds=n_seconds), (tf.float32, tf.int32))
    dataset = dataset.map(lambda x, y: (tf.squeeze(x, axis=0), y))
    dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(prefetch)
    return dataset.make_initializable_iterator()

def data_pipeline_gan(fps, batch_size, slice_len, decode_fs, decode_parallel_calls=1, slice_randomize_offset=False, slice_first_only=False, slice_overlap_ratio=0, slice_pad_end=False, repeat=False, shuffle=False, shuffle_buffer_size=None, prefetch_size=None, prefetch_gpu_num=None):

    dataset = tf.data.Dataset.from_tensor_slices(fps)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)

    if repeat:
        dataset = dataset.repeat()

    def _decode_audio_reshaped(fp, sample_rate):
        _wav = decode_audio(fp, sample_rate=sample_rate).astype(np.float32)
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

    dataset = dataset.batch(batch_size, drop_remainder=True)

    if prefetch_size is not None:
        dataset = dataset.prefetch(prefetch_size)
        if prefetch_gpu_num is not None and prefetch_gpu_num >= 0:
            dataset = dataset.apply(
                tf.data.experimental.prefetch_to_device(
                    '/device:GPU:{}'.format(prefetch_gpu_num)))

    iterator = dataset.make_one_shot_iterator()

    return iterator.get_next()