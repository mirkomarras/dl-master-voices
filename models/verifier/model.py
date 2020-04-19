#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.metrics import roc_curve
from scipy import spatial
import tensorflow as tf
import pandas as pd
import numpy as np
import os

from helpers.audio import get_tf_spectrum, get_tf_filterbanks, play_n_rec

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class StepDecay():
    def __init__(self, init_alpha=0.01, decay_factor=0.25, decay_step=10):
        self.init_alpha = init_alpha
        self.decay_factor = decay_factor
        self.decay_step = decay_step

    def __call__(self, epoch):
        exp = np.floor((1 + epoch) / self.decay_step)
        alpha = self.init_alpha * (self.decay_factor ** exp)
        print('Learning rate for next epoch', float(alpha))
        return float(alpha)

class VladPooling(tf.keras.layers.Layer):
    def __init__(self, mode, k_centers, g_centers=0, **kwargs):
        self.k_centers = k_centers
        self.g_centers = g_centers
        self.mode = mode
        super(VladPooling, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'mode': self.mode,
            'k_centers': self.k_centers,
            'g_centers': self.g_centers
        })
        return config

    def build(self, input_shape):
        self.cluster = self.add_weight(shape=[self.k_centers+self.g_centers, input_shape[0][-1]], name='centers', initializer='orthogonal')
        self.built = True

    def compute_output_shape(self, input_shape):
        assert input_shape
        return (input_shape[0][0], self.k_centers*input_shape[0][-1])

    def call(self, x):
        feat, cluster_score = x
        num_features = feat.shape[-1]

        max_cluster_score = tf.keras.backend.max(cluster_score, -1, keepdims=True)
        exp_cluster_score = tf.keras.backend.exp(cluster_score - max_cluster_score)
        A = exp_cluster_score / tf.keras.backend.sum(exp_cluster_score, axis=-1, keepdims = True)

        A = tf.keras.backend.expand_dims(A, -1)
        feat_broadcast = tf.keras.backend.expand_dims(feat, -2)
        feat_res = feat_broadcast - self.cluster
        weighted_res = tf.math.multiply(A, feat_res)
        cluster_res = tf.keras.backend.sum(weighted_res, [1, 2])

        if self.mode == 'gvlad':
            cluster_res = cluster_res[:, :self.k_centers, :]

        cluster_l2 = tf.keras.backend.l2_normalize(cluster_res, -1)
        outputs = tf.keras.backend.reshape(cluster_l2, [-1, int(self.k_centers) * int(num_features)])
        return outputs

class Model(object):
    """
       Class to represent Speaker Verification (SV) models with model saving / loading and playback & recording capabilities
    """

    def __init__(self, name='', id=-1, noises=None, cache=None, n_seconds=3, sample_rate=16000, emb_size=1024):
        """
        Method to initialize a speaker verification model that will be saved in 'data/pt_models/{name}'
        :param name:        String id for this model
        :param id:          Version id for this model - default: auto-increment value along the folder 'data/pt_models/{name}'
        :param noises:      Dictionary of paths to noise audio samples, e.g., noises['room'] = ['xyz.wav', ...]
        :param cache:       Dictionary of noise audio samples, e.g., cache['xyz.wav'] = [0.1, .54, ...]
        :param n_seconds:   Maximum number of seconds of an audio sample to be processed
        :param sample_rate: Sample rate of an audio sample to be processed
        """
        self.noises = noises
        self.cache = cache

        self.sample_rate=sample_rate
        self.n_seconds = n_seconds
        self.emb_size = emb_size

        self.name = name
        self.dir = os.path.join('.', 'data', 'pt_models', self.name)
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        self.id = len(os.listdir(self.dir)) if id < 0 else id
        if not os.path.exists(os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id)))):
            os.makedirs(os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id))))
        print('> created model folder', os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id))))

    def get_model(self):
        return self.inference_model

    def build(self, classes=None, loss='softmax', aggregation='avg', vlad_clusters=12, ghost_clusters=2, weight_decay=1e-4):
        """
        Method to build a speaker verification model that takes audio samples of shape (None, 1) and impulse flags (None, 3)
        :param classes:         Number of classes that this model should manage during training
        :param loss:            Type of loss
        :param aggregation:     Type of aggregation function
        :param vlad_clusters:   Number of vlad clusters in vlad and gvlad
        :param ghost_clusters:  Number of ghost clusters in vlad and gvlad
        :param weight_decay:    Decay of weights in convolutional layers
        :return:                None
        """
        self.model = None
        self.inference = None
        self.classes = classes

    def save(self):
        """
        Method to save the weights of this model in 'data/pt_models/{name}/v{id}/model.tf'
        :return:            None
        """
        print('>', 'saving', self.name, 'model')
        self.model.save(os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id)), 'model.h5'))
        print('>', 'saved', self.name, 'model in', os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id))))

    def load(self):
        """
        Method to load weights for this model from 'data/pt_models/{name}/v{id}/model.tf'
        :return:            None
        """
        print('>', 'loading', self.name, 'model')
        if os.path.exists(os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id)))):
            if len(os.listdir(os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id))))) > 0:
                self.model = tf.keras.models.load_model(os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id)), 'model.h5'), custom_objects={'VladPooling':VladPooling})
                self.inference_model = tf.keras.Model(inputs=self.model.input, outputs=self.model.get_layer('fc7').output)
                print('>', 'loaded model from', os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id))))
            else:
                print('>', 'no pre-trained model for', self.name, 'model from', os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id))))
        else:
            print('>', 'no directory for', self.name, 'model at', os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id))))

    def embed(self, signal):
        """
        Method to compute the embedding vector extracted by this model from signal with no playback & recording
        :param signal:      The audio signal from which the embedding vector will be extracted - shape (None,1)
        :return:            None
        """
        return self.inference_model.predict(signal)

    def train(self, train_data, val_data, noises, cache, augment=0, mode='spectrum', batch_size=32, steps_per_epoch=10, epochs=1, learning_rate=1e-1, decay_factor=0.1, decay_step=10, optimizer='adam'):
        """
        Method to train and validate this model
        :param train_data:      Training data pipeline - shape ({'input_1': (batch, None, 1), 'input_2': (batch, 3)}), (batch, classes)
        :param test_data:       Pre-computed testing data pairs - shape ((pairs, None, 1), (pairs, None, 1)), (pairs, binary_label)
        :param steps_per_epoch: Number of steps per epoch
        :param epochs:          Number of training epochs
        :param learning_rate:   Learning rate
        :param decay_factor:    Decay in terms of learning rate
        :param decay_step:      Number of epoch for each decay in learning rate
        :param optimizer:       Type of training optimizer
        :return:                None
        """

        print('>', 'training', self.name, 'model')
        original_name = self.model.name
        schedule = StepDecay(init_alpha=learning_rate, decay_factor=decay_factor, decay_step=decay_step)
        lr_callback = tf.keras.callbacks.LearningRateScheduler(schedule)

        signal_input = tf.keras.Input(shape=(None, 1,), name='Input_1')
        impulse_input = tf.keras.Input(shape=(3,), name='Input_2')

        if augment:
            print('> loading augmented model')
            x = tf.keras.layers.Lambda(lambda x: play_n_rec(x, noises, cache, batch_size), name='playback_layer')([signal_input, impulse_input])
            if mode == 'spectrum':
                signal_output = tf.keras.layers.Lambda(lambda x: get_tf_spectrum(x), name='acoustic_layer')(x)
            else:
                signal_output = tf.keras.layers.Lambda(lambda x: get_tf_filterbanks(x), name='acoustic_layer')(x)
        else:
            print('> loading not augmented model')
            if mode == 'spectrum':
                signal_output =  tf.keras.layers.Lambda(lambda x: get_tf_spectrum(x[0]), name='acoustic_layer')([signal_input, impulse_input])
            else:
                signal_output =  tf.keras.layers.Lambda(lambda x: get_tf_filterbanks(x[0]), name='acoustic_layer')([signal_input, impulse_input])

        extractor = tf.keras.models.Model(inputs=[signal_input, impulse_input], outputs=[signal_output])
        self.sup_model = tf.keras.models.Model(inputs=[signal_input, impulse_input], outputs=[self.model(extractor([signal_input, impulse_input]))])
        self.sup_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        for epoch in range(epochs):
            self.sup_model.fit(train_data, steps_per_epoch=steps_per_epoch, initial_epoch=epoch, epochs=epoch+1, callbacks=[lr_callback])
            self.model = tf.keras.models.Model(inputs=self.sup_model.get_layer(original_name).input, outputs=self.sup_model.get_layer(original_name).output)
            self.save()

        print('>', 'trained', self.name, 'model')

    def test(self, test_data, policy='any', mode='spectrum', save=False):
        """
        Method to test this model against verification attempts
        :param test_data:       Pre-computed testing data pairs - shape ((pairs, None, 1), (pairs, None, 1)), (pairs, binary_label)
        :return:                (Model EER, EER threshold, FAR1% threshold)
        """
        print('>', 'testing', self.name, 'model on policy', policy)
        (x1, x2), y = test_data
        eer, thr_eer, id_eer, thr_far1, id_far1 = 0, 0, 0, 0, 0
        far, frr = [], []
        similarity_scores = []
        target_scores = []
        for pair_id, (f1, f2, label) in enumerate(zip(x1, x2, y)):
            inp_1 = get_tf_spectrum(f1) if mode == 'spectrum' else get_tf_filterbanks(f1)
            inp_2 = [get_tf_spectrum(f) if mode == 'spectrum' else get_tf_filterbanks(f) for f in (f2 if isinstance(f2, list) else [f2])]
            emb1 = tf.keras.layers.Lambda(lambda emb1: tf.keras.backend.l2_normalize(emb1, 1))(self.embed(inp_1))
            emb2 = [tf.keras.layers.Lambda(lambda emb2: tf.keras.backend.l2_normalize(emb2, 1))(self.embed(inp)) for inp in inp_2]
            target_scores.append(label)
            similarity_scores.append(tf.keras.layers.Dot(axes=1, normalize=True)([emb1, np.mean(emb2, axis=0)])[0] if policy == 'avg' else np.max([tf.keras.layers.Dot(axes=1, normalize=True)([emb1, emb])[0] for emb in emb2]))
            if pair_id > 2:
                far, tpr, thresholds = roc_curve(target_scores, similarity_scores, pos_label=1)
                frr = 1 - tpr
                id_eer = np.argmin(np.abs(far - frr))
                id_far1 = np.argmin(np.abs(far - 0.01))
                eer = float(np.mean([far[id_eer], frr[id_eer]]))
                thr_eer = thresholds[id_eer]
                thr_far1 = thresholds[id_far1]
                print('\r> pair', pair_id+1, 'of', len(x1), '- eer', round(eer, 4), 'thr@eer', round(thr_eer, 4), 'thr@far1', round(thr_far1, 4), end='')
        print('\n>', 'tested', self.name, 'model')
        if save:
            df = pd.DataFrame({'target': target_scores, 'similarity': similarity_scores})
            df.to_csv(os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id)), 'test_results_' + str(round(eer, 4)) + '.csv'))
            print('>', 'saved results in', os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id)), 'test_results.csv'))
        return (eer, far[id_eer], frr[id_eer], thr_eer), (far[id_far1], frr[id_far1], thr_far1)

    def impersonate(self, impostor_signal, threshold, policy, x_mv_test, y_mv_test, male_x_mv_test, female_x_mv_test, n_templates=10):
        """
        Method to test this model under impersonation attempts
        :param impostor_signal:     Audio signal against which this model is tested - shape (None, 1)
        :param threshold:           Verification threshold
        :param policy:              Verification policy - choices ['avg', 'any']
        :param x_mv_test:           Testing users' audio samples - shape (users, n_templates, None, 1)
        :param y_mv_test:           Testing users' labels - shape (users, n_templates)
        :param male_x_mv_test:      Male users' ids
        :param female_x_mv_test:    Female users' ids
        :param n_templates:         Number of audio samples to create a user template
        :return:                    {'m': impersonation rate against male users, 'f': impersonation rate against female users}
        """

        print('>', 'impersonating', self.name, 'model')
        mv_emb = tf.keras.layers.Lambda(lambda emb1: tf.keras.backend.l2_normalize(emb1, 1))(self.embed(impostor_signal))
        mv_fac = np.zeros(len(np.unique(y_mv_test)))
        for class_index, class_label in enumerate(np.unique(y_mv_test)):
            template = [self.embed(signal) for signal in x_mv_test[class_index*n_templates:(class_index+1)*n_templates]]
            if policy == 'any':
                mv_fac[class_index] = len([1 for template_emb in np.array(template) if tf.keras.layers.Dot(axes=1, normalize=True)([template_emb, mv_emb]) > threshold])
            elif policy == 'avg':
                mv_fac[class_index] = 1 if tf.keras.layers.Dot(axes=1, normalize=True)([mv_emb, np.mean(np.array(template), axis=0)]) else 0
        print('>', 'impersonated', self.name, 'model')
        return {'m': mv_fac[np.array(male_x_mv_test)], 'f': mv_fac[np.array(female_x_mv_test)]}