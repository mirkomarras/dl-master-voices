#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scipy.spatial.distance import euclidean, cosine
from sklearn.metrics import roc_curve
import tensorflow as tf
import pandas as pd
import numpy as np
import os

from helpers.audio import get_tf_spectrum, get_tf_filterbanks

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
        print('Learning rate for next epoch', float(alpha))
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


    def __init__(self, name='', id=-1):
        '''
        An Automated Speaker Verification (ASV) model
        :param name:        Name of the model
        :param id:          Model instance ID
        '''

        self.name = name

        self.dir = os.path.join('.', 'data', 'pt_models', self.name)
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

        self.id = len(os.listdir(self.dir)) if id < 0 else id
        if not os.path.exists(os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id)))):
            os.makedirs(os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id))))

        print('> created model folder', os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id))))


    def infer(self):
        '''
        Create a model instance ready to generate speaker embeddings
        :return: Inference model
        '''
        return tf.keras.Model(inputs=self.model.input, outputs=self.model.get_layer(self.embs_name).output)


    def build(self, classes=0, embs_size=512, embs_name='embs', loss='softmax', aggregation='avg', vlad_clusters=12, ghost_clusters=2, weight_decay=1e-3, mode='train'):
        '''

        Method to build a speaker verification model that takes audio samples of shape (None, 1) and impulse flags (None, 3)
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
        print('>', 'saving', self.name, 'model')
        self.model.save(os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id)), 'model.h5'))
        pd.DataFrame(self.history, columns=['loss', 'acc', 'err', 'far@eer', 'frr@eer', 'thr@eer', 'far@far1', 'frr@far1', 'thr@far']).to_csv(os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id)), 'history.csv'), index=False)
        print('>', 'saved', self.name, 'model in', os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id))))


    def load(self):
        """
        Load this model
        """
        print('>', 'loading', self.name, 'model')
        if os.path.exists(os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id)))):
            if len(os.listdir(os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id))))) > 0:
                self.model = tf.keras.models.load_model(os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id)), 'model.h5'), custom_objects={'VladPooling': VladPooling})
                self.history = []
                if os.path.exists(os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id)), 'history.csv')):
                    self.history = pd.read_csv(os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id)), 'history.csv')).values.tolist()
                print('>', 'loaded model from', os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id))))
            else:
                print('>', 'no pre-trained model for', self.name, 'model from', os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id))))
        else:
            print('>', 'no directory for', self.name, 'model at', os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id))))


    def train(self, train_data, val_data, output_type='spectrum', steps_per_epoch=10, epochs=1024, learning_rate=1e-3, decay_factor=0.1, decay_step=10, optimizer='adam'):
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

        lr_callback = tf.keras.callbacks.LearningRateScheduler(StepDecay(init_alpha=learning_rate, decay_factor=decay_factor, decay_step=decay_step))

        self.model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        for epoch in range(epochs):
            tf_history = self.model.fit(train_data, steps_per_epoch=steps_per_epoch, initial_epoch=epoch, epochs=epoch + 1, callbacks=[lr_callback]).history
            self.history.append([tf_history['loss'][0], tf_history['accuracy'][0]] + self.test(val_data, output_type))
            self.save()

        print('>', 'trained', self.name, 'model')


    def test(self, test_data, output_type='spectrum', policy='any', save=False):
        """
        Method to test this model against verification attempts
        :param test_data:       Pre-computed testing data pairs - shape ((pairs, None, 1), (pairs, None, 1)), (pairs, binary_label)
        :return:                (Model EER, EER threshold, FAR1% threshold)
        """
        print('>', 'testing', self.name, 'model on policy', policy)

        (x1, x2), y = test_data

        scores = []
        labels = []
        extractor = self.infer()
        eer, thr_eer, id_eer, thr_far1, id_far1, far, frr = 0, 0, 0, 0, 0, [], []
        for pair_id, (f1, f2, label) in enumerate(zip(x1, x2, y)):
            inp_1 = get_tf_spectrum(f1, num_fft=512) if output_type == 'spectrum' else get_tf_filterbanks(f1, n_filters=24)
            inp_2 = [get_tf_spectrum(f, num_fft=512) if output_type == 'spectrum' else get_tf_filterbanks(f, n_filters=24) for f in (f2 if isinstance(f2, list) else [f2])]
            emb1 = tf.keras.layers.Lambda(lambda emb1: tf.keras.backend.l2_normalize(emb1, 1))(extractor.predict(inp_1))
            emb2 = [tf.keras.layers.Lambda(lambda emb2: tf.keras.backend.l2_normalize(emb2, 1))(extractor.predict(inp)) for inp in inp_2]

            labels.append(label)
            scores.append(1 - cosine(emb1, np.mean(emb2, axis=0)) if policy == 'avg' else np.max([1 - cosine(emb1, emb) for emb in emb2]))

            if (pair_id + 1) % 5 == 0 and pair_id > 0:
                far, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
                frr = 1 - tpr
                id_eer = np.argmin(np.abs(far - frr))
                id_far1 = np.argmin(np.abs(far - 0.01))
                eer = float(np.mean([far[id_eer], frr[id_eer]]))
                thr_eer = thresholds[id_eer]
                thr_far1 = thresholds[id_far1]
                print('\r> pair', pair_id + 1, 'of', len(x1), '- eer', round(eer, 4), 'thr@eer', round(thr_eer, 4), 'thr@far1', round(thr_far1, 4), end='')

        print('\n>', 'tested', self.name, 'model')

        if save:
            df = pd.DataFrame({'target': scores, 'similarity': labels})
            df.to_csv(os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id)), 'test_vox1_sv_test.csv'))
            print('>', 'saved results in', os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id)), 'test_vox1_sv_test.csv'))

        return [eer, far[id_eer], frr[id_eer], thr_eer, far[id_far1], frr[id_far1], thr_far1]


    def impersonate(self, input_mv, thresholds, x_mv_test_embs, y_mv_test, male_x_mv_test, female_x_mv_test, n_templates=10, mode='spectrum'):
        """
        Method to test this model under impersonation attempts
        :param input_mv:            Input spectrogram against which this model is tested - shape (None, 1)
        :param thresholds:          List of verification threshold
        :param policy:              Verification policy - choices ['avg', 'any']
        :param x_mv_test_embs:      Testing users' embeddings - shape (users, n_templates, 512)
        :param y_mv_test:           Testing users' labels - shape (users, n_templates)
        :param male_x_mv_test:      Male users' ids
        :param female_x_mv_test:    Female users' ids
        :param n_templates:         Number of audio samples to create a user template
        :return:                    {'m': impersonation rate against male users, 'f': impersonation rate against female users}
        """

        # We extract the speaker embedding associated to the master voice input
        extractor = self.infer()
        mv_emb = tf.keras.layers.Lambda(lambda emb1: tf.keras.backend.l2_normalize(emb1, 1))(extractor.predict(np.expand_dims(input_mv, axis=0)))
        scores = [1 - cosine(mv_emb, emb) for emb in x_mv_test_embs]

        # We set up an array of shape no_thresholds x no_test_user where we count the false accepts for the input master vice against the current user with the current thresholds
        mv_fac = np.zeros((len(thresholds), len(np.unique(y_mv_test))))

        for class_index, class_label in enumerate(np.unique(y_mv_test)): # For each user in the test set
            # We extract the enrolled embeddings for the current user
            user_scores = scores[class_index*n_templates:(class_index+1)*n_templates]
            for thr_index, threshold in enumerate(thresholds): # For each verification threshold
                mv_fac[thr_index, class_index] = min(1, len([1 for score in user_scores if score > threshold]))

        results = []
        for thr_index, _ in enumerate(thresholds): # For each threshold, we separately compute the percentage of females (males) users who have been impersonated
            results.append({'m': np.sum(mv_fac[thr_index, np.array(male_x_mv_test)]) / len(male_x_mv_test), 'f': np.sum(mv_fac[thr_index,np.array(female_x_mv_test)]) / len(female_x_mv_test)})

        return results