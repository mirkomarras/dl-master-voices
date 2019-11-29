#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.metrics import roc_curve, auc
from scipy import spatial
import tensorflow as tf
import numpy as np
import random
import time
import os

from src.helpers.audio import decode_audio

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Model(object):

    def __init__(self, graph=tf.Graph(), var2std_epsilon=0.00001, reuse=False, id=''):
        self.graph = graph
        self.var2std_epsilon = var2std_epsilon
        self.reuse = reuse
        self.noises = None
        self.cache = None
        self.id = id

        self.frame_size=0.025
        self.frame_stride=0.01
        self.num_fft=512
        self.lower_edge_hertz=80.0
        self.upper_edge_hertz=8000.0

        self.n_filters=24
        self.sample_rate=16000
        self.n_seconds = 3

    def get_version_id(self, id=''):
        tf_dir = os.path.join('.', 'data', 'pt_models', self.name, 'tf')
        tf_v = str(len(os.listdir(tf_dir))) if not id else id
        out_dir = os.path.join(tf_dir, 'v' + tf_v)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        assert os.path.exists(os.path.join(out_dir))
        self.dir = out_dir
        return tf_v

    def build(self, input_x, input_y, n_classes=0, n_filters=24, noises=None, cache=None, augment=0, n_seconds=3, sample_rate=16000):
        self.n_classes = n_classes
        self.noises = noises
        self.cache = cache

        self.sample_rate = sample_rate
        self.n_seconds = n_seconds
        self.n_filters = n_filters

        # Placeholder for parameter
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.phase = tf.placeholder(tf.bool, name='phase')

        # Placeholders for regular data
        self.input_x = input_x if input_x is not None else tf.placeholder(tf.float32, [None, self.sample_rate*self.n_seconds, 1], name='input_x')
        self.input_y = input_y if input_y is not None else tf.placeholder(tf.int32, [None], name='input_y')

        # Computation for playback-and-recording
        self.speaker = tf.placeholder(tf.float32, [None, 1, 1], name='speaker')
        self.room = tf.placeholder(tf.float32, [None, 1, 1], name='room')
        self.microphone = tf.placeholder(tf.float32, [None, 1, 1], name='microphone')

        noise_strength = tf.clip_by_value(tf.random.normal((1,), 0, 5e-3), 0, 10)
        speaker_out = tf.nn.conv1d(self.input_x, self.speaker, 1, padding="SAME")
        noise_tensor = tf.random.normal(tf.shape(self.input_x), mean=0, stddev=noise_strength, dtype=tf.float32)
        speaker_out = tf.add(speaker_out, noise_tensor)
        room_out = tf.nn.conv1d(speaker_out, self.room, 1, padding="SAME")
        self.input_a = tf.nn.conv1d(room_out, self.microphone, 1, padding='SAME', name='input_a') if augment else self.input_x

    def save(self, sess, out_dir=''):
        print('>', 'saving', self.name, 'graph')
        if not out_dir:
            out_dir = self.dir
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        assert os.path.exists(os.path.join(out_dir))
        saver = tf.train.Saver()
        out_path = saver.save(sess, os.path.join(out_dir, 'model'))
        print('>', self.name, 'graph saved in path', out_path)

    def load(self, sess, in_dir='', verbose=1):
        if verbose:
            print('>', 'loading', self.name, 'graph')
        if not in_dir:
            in_dir = self.dir
        assert os.path.exists(in_dir)
        saver = tf.train.import_meta_graph(os.path.join(in_dir, 'model.meta'))
        saver.restore(sess, os.path.join(in_dir, 'model'))
        self.graph = sess.graph
        if verbose:
            print('>', self.name, 'graph restored from path', in_dir)

    def embed(self, sess, x):
        feed_dict = {self.input_a: np.reshape(x, (1, x.shape[0], x.shape[1])),
                     self.speaker: np.zeros(1).reshape((-1, 1, 1)),
                     self.room: np.zeros(1).reshape((-1, 1, 1)),
                     self.microphone: np.zeros(1).reshape((-1, 1, 1)),
                     self.dropout_keep_prob: 1.0,
                     self.phase: False}
        return sess.run(self.embedding, feed_dict=feed_dict)

    def train(self, n_epochs, n_steps_per_epoch, learning_rate, dropout_proportion, initializer, validation_data=None, validation_interval=1, print_interval=1):
        assert n_epochs > 0 and n_steps_per_epoch > 0 and print_interval > 0

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=False, log_device_placement=False)) as sess:
            sess.run(tf.global_variables_initializer())

            print('>', 'training', self.name, 'model')
            for epoch in range(n_epochs):
                print('>', 'epoch', epoch+1, '/', n_epochs)
                sess.run(initializer)

                batch_count = n_steps_per_epoch
                dropout_keep_prob = 1 - dropout_proportion

                total_gpu_waiting = 0.0
                total_loss, batch_loss = 0, 0
                total_accuracy, batch_accuracy = 0, 0

                for batch_id in range(batch_count):

                    f_speaker = random.sample(self.noises['speaker'], 1)[0]
                    f_room = random.sample(self.noises['room'], 1)[0]
                    f_mic = random.sample(self.noises['microphone'], 1)[0]

                    feed_dict = {
                        self.speaker: self.cache[f_speaker],
                        self.room: self.cache[f_room],
                        self.microphone: self.cache[f_mic],
                        self.dropout_keep_prob: dropout_keep_prob,
                        self.learning_rate: learning_rate,
                        self.phase: True
                    }

                    gpu_waiting = time.time()
                    _, loss, accuracy = sess.run([self.optimizer, self.loss, self.accuracy], feed_dict=feed_dict)
                    curr_gpu_waiting = time.time() - gpu_waiting
                    total_gpu_waiting += curr_gpu_waiting

                    total_loss += loss
                    total_accuracy += accuracy

                    if batch_id % print_interval == 0:
                        print('\r>', 'step %4.0f / %4.0f - eta: %4.0fm - loss: %3.5f - acc: %3.5f - time/step: %3.1fs' % (batch_id+1, batch_count, (batch_count - (batch_id+1)) * curr_gpu_waiting // 60, total_loss/(batch_id+1), total_accuracy/(batch_id+1), curr_gpu_waiting), end='')

                if validation_data and (epoch+1) % validation_interval == 0:
                    print()
                    self.validate(sess, validation_data)

                print()
                self.save(sess)

            print('>', 'trained', self.name, 'model')

    def validate(self, sess, validation_data, comp_func=lambda x, y: 1 - spatial.distance.cosine(x, y), print_interval=100):
        (x1, x2), y = validation_data

        thr_eer, thr_far1 = 0, 0
        s = np.zeros(len(x1))
        for id, (f1, f2) in enumerate(zip(x1, x2)):
            emb_1 = self.embed(sess, f1)
            emb_2 = self.embed(sess, f2)
            s[id] = comp_func(emb_1, emb_2)

            if (id+1) % print_interval == 0 or (id+1) == len(x1):
                far, tpr, thresholds = roc_curve(y[:id+1], s[:id+1])
                frr = 1 - tpr

                id_eer = np.argmin(np.abs(far - frr))
                id_far1 = np.argmin(np.abs(far - 0.01))

                eer = float(np.mean([far[id_eer], frr[id_eer]]))
                thr_eer = thresholds[id_eer]
                thr_far1 = thresholds[id_far1]

                print('\r> pair %5.0f / %5.0f - eer: %3.5f - thr@eer: %3.5f - thr@far1: %3.1f' % (id+1, len(x1), eer, thr_eer, thr_far1), end='')

        print()

        return thr_eer, thr_far1

    def test(self, test_data):
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            self.load(sess, self.dir)

            print('Testing', self.name, 'model')
            return self.validate(sess, test_data)

    def impersonate(self, mv_data, threshold, policy, x_mv_test, y_mv_test, male_x_mv_test, female_x_mv_test, n_templates=10, comp_func=lambda x, y: 1 - spatial.distance.cosine(x, y)):

        tf_dir = os.path.join('.', 'data', 'pt_models', self.name, 'tf')
        tf_v = self.id
        in_dir = os.path.join(tf_dir, 'v' + tf_v)

        assert os.path.exists(in_dir)

        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            self.load(sess, in_dir, 0)

            mv_emb = self.embed(sess, mv_data)
            mv_fac = np.zeros(len(np.unique(y_mv_test)))

            for class_index, class_label in enumerate(np.unique(y_mv_test)):
                template = [self.embed(sess, signal) for signal in x_mv_test[class_index*n_templates:(class_index+1)*n_templates]]
                if policy == 'any':
                    mv_fac[class_index] = len([1 for template_emb in np.array(template) if comp_func(template_emb, mv_emb) > threshold])
                elif policy == 'avg':
                    mv_fac[class_index] = 1 if comp_func(mv_emb, np.mean(np.array(template), axis=0)) > threshold else 0
                else:
                    raise NotImplementedError('Verification policy not implemented')

            return {'m': len([index for index, fac in enumerate(mv_fac) if fac > 0 and index in male_x_mv_test]) / len(male_x_mv_test), 'f': len([index for index, fac in enumerate(mv_fac) if fac > 0 and index in female_x_mv_test]) / len(female_x_mv_test)}