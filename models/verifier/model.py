#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scipy.spatial.distance import euclidean, cosine
from sklearn.metrics import roc_curve
from itertools import groupby
from loguru import logger
import tensorflow as tf
from tqdm import tqdm
import pandas as pd
import numpy as np
import json
import os

from helpers.audio import load_noise_paths, cache_noise_data, get_play_n_rec_audio
from helpers.audio import get_tf_spectrum, get_tf_filterbanks, decode_audio
from helpers.dataset import Dataset

from loguru import logger

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class StepDecay():
    def __init__(self, init_alpha=0.01, decay_factor=0.25, decay_step=10):
        '''
        Implementation of the learning rate decay along epochs
        :param init_alpha:      Initial learning rate
        :param decay_factor:    Decay factor
        :param decay_step:      Decay step as a number of epochs
        '''
        self.init_alpha = init_alpha
        self.decay_factor = decay_factor
        self.decay_step = decay_step

    def __call__(self, epoch):
        exp = np.floor((1 + epoch) / self.decay_step)
        alpha = self.init_alpha * (self.decay_factor ** exp)
        # logger.debug('Learning rate for next epoch', float(alpha))
        return float(alpha)


class VladPooling(tf.keras.layers.Layer):

    def __init__(self, mode, k_centers, g_centers=0, **kwargs):
        '''
        Implementation of the VLAD pooling layers

        # References:
        [1] NetVLAD: CNN architecture for weakly supervised place recognition. https://arxiv.org/pdf/1511.07247.pdf
        [2] Ghostvlad for set-based face recognition. https://arxiv.org/pdf/1810.09951.pdf

        :param mode:        Type of polling layer: 'vlad' or 'gvlad'
        :param k_centers:   Number of centroids
        :param g_centers:   Number of ghost centroids
        '''
        self.k_centers = k_centers
        self.g_centers = g_centers
        self.mode = mode
        super().__init__(**kwargs)

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


    def __init__(self, name='', id=-1):
        '''
        An Automated Speaker Verification (ASV) model

        # Typical way of setting up the model

        Either:
        | sv = xvector.XVector(id=0)
        | sv = verifier.get_model(net)

        sv.build(classes=0, mode='test')
        sv.load()
        sv.calibrate_thresholds()
        sv.infer()

        :param name:        Name of the model
        :param id:          Model instance ID
        '''

        # TODO Simplify model setup - current split into build+load+calibrate+infer is unnecessarily complicated
        self.name = name
        self.input_type = None
        self._inference_model = None
        self._thresholds = None
        self._uses_spectrum = True

        self._noise_paths = None
        self._noise_cache = None
        self.impulse_flags = (1, 1, 1)

        self.dir = os.path.join('.', 'data', 'vs_mv_models', self.name)
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

        self.id = len(os.listdir(self.dir)) if id < 0 else id
        if not os.path.exists(os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id)))):
            os.makedirs(os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id))))

        path = os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id)))
        logger.info('created model folder {}'.format(path))


    @property
    def noise_paths(self):
        return self._noise_paths

    @property
    def noise_cache(self):
        return self._noise_cache

    def setup_playback(self, dirname='data/vs_noise_data', impulse_flags=(1,1,1)):
        self.impulse_flags = impulse_flags
        if dirname is None:
            self._noise_paths = None
            self._noise_cache = None
        elif self._noise_cache is None:
            self._noise_paths = load_noise_paths(dirname)
            self._noise_cache = cache_noise_data(self._noise_paths, sample_rate=16000)

    def infer(self):
        '''
        Create a model instance ready to generate speaker embeddings
        :return: Inference model
        '''
        if self._inference_model is None:
            self._inference_model = tf.keras.Model(inputs=self.model.input, outputs=self.model.get_layer(self.embs_name).output)
        return self._inference_model


    def build(self, classes=0, embs_name='embs', embs_size=512, loss='softmax', aggregation='avg', vlad_clusters=10, ghost_clusters=2, weight_decay=1e-3, mode='train'):
        '''
        
        Abstract method to build a speaker verification model. Should be implemented in child classes.

        The model should take audio samples of shape (None, 1) as input and have an internal layer
        <embs_name> that represents the speaker embedding. Most models are pre-trained on a speaker 
        classification problem.

        :param classes:         Number of classes that this model should manage during training
        :param embs_size:       Size of the speaker embedding vector to be returned by the model
        :param embs_name:       Name of the layer from which embeddings are extracted
        :param loss:            Type of loss
        :param aggregation:     Type of aggregation function
        :param vlad_clusters:   Number of vlad clusters in vlad and gvlad
        :param ghost_clusters:  Number of ghost clusters in vlad and gvlad
        :param weight_decay:    Decay of weights in convolutional layers
        :param mode:            Building mode between 'train' and 'test'
        :return:
        '''
        self.history = []
        self.model = None
        self.embs_size = embs_size
        self.embs_name = embs_name


    def save(self):
        """
        Save this model
        """
        logger.info('saving {self.name} model')
        path = os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id)))
        self.model.save(os.path.join(path, 'model.h5'))
        pd.DataFrame(self.history, columns=['loss', 'acc', 'err', 'far@eer', 'frr@eer', 'thr@eer', 'far@far1', 'frr@far1', 'thr@far']).to_csv(os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id)), 'history.csv'), index=False)
        logger.info('saved {} model in {}'.format(self.name, path))


    def save_params(self, params):
        path = os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id)), 'params.txt')
        with open(path, "w") as file:
            for arg in vars(params):
                file.write("%s,%s\n" % (arg, getattr(params, arg)))
        logger.info('params saved in {}'.format(path))


    def get_dirname(self):
        return os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id)))


    def load(self, replace_model=False):
        """
        Load this model
        """
        logger.info('loading pre-trained {}'.format(self.name))
        version_id = 'v' + str('{:03d}'.format(self.id))
        if os.path.exists(os.path.join(self.dir, version_id)):
            if len(os.listdir(os.path.join(self.dir, version_id))) > 0:
                model_path = os.path.join(self.dir, version_id, 'model.h5')
                logger.debug(model_path)
                if replace_model:
                    self.model = tf.keras.models.load_model(model_path, custom_objects={'VladPooling': VladPooling})
                else:
                    self.model.load_weights(model_path, skip_mismatch=True, by_name=True)
                self.history = []
                if os.path.exists(os.path.join(self.dir, version_id, 'history.csv')):
                    self.history = pd.read_csv(os.path.join(self.dir, version_id, 'history.csv')).values.tolist()
                
                if replace_model:
                    logger.info('loaded model from {}'.format(os.path.join(self.dir, version_id)))
                else:
                    logger.info('loaded weights from {}'.format(os.path.join(self.dir, version_id)))

            else:
                logger.warning('No pre-trained model for {} in {}'.format(self.name, os.path.join(self.dir, version_id)))
        else:
            logger.error('No directory for {} model at {}'.format(self.name, os.path.join(self.dir, version_id)))


    def train(self, train_data, val_data, steps_per_epoch=10, epochs=1024, learning_rate=1e-3, decay_factor=0.1, decay_step=10, optimizer='adam'):
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

        logger.info('Starting training for {}'.format(self.name))

        lr_callback = tf.keras.callbacks.LearningRateScheduler(StepDecay(init_alpha=learning_rate, decay_factor=decay_factor, decay_step=decay_step))

        self.model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        for epoch in range(epochs):
            tf_history = self.model.fit(train_data, steps_per_epoch=steps_per_epoch, initial_epoch=epoch, epochs=epoch + 1, callbacks=[lr_callback]).history
            self.history.append([tf_history['loss'][0], tf_history['accuracy'][0]])
            self.save()

        logger.info('Training finished for {}'.format(self.name))


    def compute_acoustic_representation(self):
        pass

    def prepare_input(self, elements, playback):
        assert len(elements) > 0

        if isinstance(elements, str):
            elements = decode_audio(elements).reshape((1, -1))

        if len(elements.shape) == 1:
            if playback == 1:                
                # @tf.function
                def forward(signal, impulse_flags):
                  return get_play_n_rec_audio(signal=signal, noises=self.noise_paths, 
                                              cache=self.noise_cache, noise_strength='random', 
                                              impulse_flags=impulse_flags)
                
                # xf = forward(xt, xn).numpy()
                elements = [forward(decode_audio(e).reshape((1, -1, 1)).astype(np.float32), self.impulse_flags).numpy() for e in elements]
                elements = [np.squeeze(e) for e in elements]
                
                # elements = [playback.simulate(e) for e in elements]
            else:
                elements = [decode_audio(e) for e in elements]
        
        if len(elements[0].shape) == 1:
            elements = [self.compute_acoustic_representation(tf.reshape(tf.convert_to_tensor(e), (1, -1, 1))).numpy() for e in elements]
        
        return elements


    def predict(self, elements, playback=None):
        elements = self.prepare_input(elements, playback)
        if isinstance(elements, list):
            embeddings = tf.concat([self._inference_model.predict(e) for e in elements if not e.shape[2] == 1], axis=0)
        else:
            embeddings = self._inference_model.predict(elements)
        embeddings_norm = tf.keras.backend.l2_normalize(embeddings, axis=1)
        # tf.squeeze(embeddings_norm, axis=1)
        return embeddings_norm


    def compare(self, x1, x2, only_scores=False):
        assert self._inference_model is not None
        scores = []
        for e1, e2 in zip(x1, x2):
            if only_scores:
                emb_1 = e1
                emb_2 = e2
            else:
                emb_1 = self.predict(np.array([e1]))[0]
                emb_2 = self.predict(np.array([e2]))[0]
            scores.append(1 - cosine(emb_1, emb_2))
        return scores

    # TODO [Critical] Hard-coded paths for testing
    def calibrate_thresholds(self, comparison_data=None):
        if self._thresholds is None:

            thresholds_path = os.path.join(self.get_dirname(), 'thresholds.json')

            if os.path.exists(thresholds_path):
                with open(thresholds_path, 'r') as thresholds_file:
                    self._thresholds = json.load(thresholds_file)

            else:

                if comparison_data is None:
                    test_pairs = pd.read_csv(os.path.join('.', 'data', 'vs_mv_pairs', 'trial_pairs_vox1_test.csv'), delimiter=' ', names=['y', 'x1', 'x2'])
                    x1 = test_pairs['x1'].apply(lambda x: os.path.join('data/voxceleb1/test', x)), 
                    x2 = test_pairs['x2'].apply(lambda x: os.path.join('data/voxceleb1/test', x))
                    y = test_pairs['y']
                else:
                    x1, x2, y = comparison_data

                self.infer()
                scores = self.compare(x1, x2)

                far, tpr, thresholds = roc_curve(y, scores, pos_label=1)
                frr = 1 - tpr
                id_eer = np.argmin(np.abs(far - frr))
                id_far1 = np.argmin(np.abs(far - 0.01))
                eer = float(np.mean([far[id_eer], frr[id_eer]]))  # p = None --> EER, 1, 0.1
                thrs = {'eer': thresholds[id_eer], 'far1': thresholds[id_far1]}
                logger.info('>', 'found thresholds {} - eer of {}'.format(thrs, eer))

                with open(thresholds_path, 'w') as thresholds_file:
                    logger.info('>', 'thresholds saved in {}'.format(thresholds_path))
                    json.dump(thrs, thresholds_file)

                self._thresholds = thrs


    def test_error_rates(self, elements, gallery, policy='any', level='far1', playback=None):
        # TODO Add parameter to control the number of enrolled samples to test
        assert self._thresholds is not None and self._inference_model is not None
        # logger.info('used thresholds {}'.format(self._thresholds))

        # Expand the elements so that we can flexibly manage sequences of elements
        elements = elements if len(elements.shape) >= 2 else np.expand_dims(elements, axis=0)  # audio (None,) audios (None, n) ---> use prepare_batch

        # Initialize the similarity matrix (elements, utterances) for any and (elements, users) for avg --> check whether
        sim_matrix = np.zeros((len(elements), len(gallery.user_ids))) if policy == 'any' else np.zeros((len(elements), len(np.unique(gallery.user_ids))))
        # Initialize the binary impersonation matrix (0: no imp, 1:imp) of shape (elements, users)
        imp_matrix = np.zeros((len(elements), len(np.unique(gallery.user_ids))), np.uint8)
        # Initialize the counting impersonation matrix for genders of shape (elements, 2) - male and females
        gnd_matrix = np.zeros((len(elements), 2))

        n_samples_pp = gallery.n_samples_per_person
        gnds_idx = gallery.user_genders[::n_samples_pp]

        for element_idx, element in enumerate(elements):
            if hasattr(element, 'numpy'):
                element = element.numpy()
            
            element_emb = self.predict(element[tf.newaxis, ...], playback) if (len(element.shape) > 1 or element.size != self.embs_size) else element
            for user_idx, user_id in enumerate(gallery.user_ids[::n_samples_pp]): # For each user in the gallery, reuse if the gallery size increase
                
                if policy == 'any':
                    user_sim = self.compare(np.tile(element_emb, (np.sum(gallery.user_ids == user_id), 1)), gallery.embeddings[gallery.user_ids == user_id], only_scores=True)
                    user_sim = np.array(user_sim)
                    sim_matrix[element_idx, gallery.user_ids == user_id] = user_sim
                    imp_matrix[element_idx, user_idx] = np.any(user_sim > self._thresholds[level])
                elif policy == 'avg':
                    user_embedding = np.mean(gallery.embeddings[gallery.user_ids == user_id], axis=0)
                    user_sim = self.compare(np.expand_dims(element_emb, axis=0), np.expand_dims(user_embedding, axis=0), only_scores=True)[0]
                    sim_matrix[element_idx, user_idx] = user_sim
                    imp_matrix[element_idx, user_idx] = user_sim > self._thresholds[level]

            gnd_matrix[element_idx, 0] = np.mean(imp_matrix[element_idx, gnds_idx == 'm'])
            gnd_matrix[element_idx, 1] = np.mean(imp_matrix[element_idx, gnds_idx == 'f'])

        return (sim_matrix, imp_matrix, gnd_matrix)