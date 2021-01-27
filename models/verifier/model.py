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
        self._inference_model = None

        self.dir = os.path.join('.', 'data', 'vs_mv_models', self.name)
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
        if self._inference_model is None:
            self._inference_model = tf.keras.Model(inputs=self.model.input, outputs=self.model.get_layer(self.embs_name).output)
        return self._inference_model


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


    def save_params(self, params):
        with open(os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id)), 'params.txt'), "w") as file:
            for arg in vars(params):
                file.write("%s,%s\n" % (arg, getattr(params, arg)))
        print('> params saved in', os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id)), 'params.txt'))


    def get_dirname(self):
        return os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id)))


    def load(self):
        """
        Load this model
        """
        print('>', 'loading', self.name, 'model')
        if os.path.exists(os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id)))):
            if len(os.listdir(os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id))))) > 0:
                model_path = os.path.join(self.dir, 'v' + f'{self.id:03d}', 'model.h5')
                self.model = tf.keras.models.load_model(model_path, custom_objects={'VladPooling': VladPooling})
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


    def test(self, test_data, output_type='spectrum', policy='any', save=False, filename='vox1'):
        """
        Test speaker verification model

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
            df.to_csv(os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id)), 'scores_' + filename + '_test.csv'))
            print('>', 'saved results in', os.path.join(self.dir, 'v' + str('{:03d}'.format(self.id)), 'scores_' + filename + '_test.csv'))

        return [eer, far[id_eer], frr[id_eer], thr_eer, far[id_far1], frr[id_far1], thr_far1]

    def test_impersonation(self, input_mv, thresholds, test_data, mode='spectrum'):
        """
        Test impersonation rate for a given speech example.

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

        test_data.embeddings, test_data.user_ids, test_data.user_ids_male, test_data.user_ids_female, n_templates

        # We extract the speaker embedding associated to the master voice input
        extractor = self.infer()
        mv_emb = extractor.predict(input_mv[tf.newaxis, ...])
        # mv_emb = tf.keras.layers.Lambda(lambda emb1: tf.keras.backend.l2_normalize(emb1, 1))(extractor.predict(np.expand_dims(input_mv, axis=0)))
        mv_emb = tf.keras.backend.l2_normalize(mv_emb, 1)
        scores = [1 - cosine(mv_emb, emb) for emb in test_data.embeddings]

        # We set up an array of shape no_thresholds x no_test_user where we count the false accepts for the input master vice against the current user with the current thresholds
        mv_fac = np.zeros((len(thresholds), len(np.unique(test_data.user_ids))))

        for class_index, class_label in enumerate(np.unique(test_data.user_ids)): # For each user in the test set
            # We extract the enrolled embeddings for the current user
            user_scores = scores[class_index*test_data.n_enrolled_examples:(class_index+1)*test_data.n_enrolled_examples]
            for thr_index, threshold in enumerate(thresholds): # For each verification threshold
                mv_fac[thr_index, class_index] = min(1, len([1 for score in user_scores if score > threshold]))

        results = []
        for thr_index, _ in enumerate(thresholds): # For each threshold, we separately compute the percentage of females (males) users who have been impersonated
            results.append({
                'm': np.sum(mv_fac[thr_index, np.array(test_data.user_ids_male)]) / len(test_data.user_ids_male),
                'f': np.sum(mv_fac[thr_index, np.array(test_data.user_ids_female)]) / len(test_data.user_ids_female)
            })

        return results

    def test_imp_extended(self):

        # if row['path1'] in speaker_embs:  # If we already computed the embedding for the first element of the verification pair
        #     emb_1 = speaker_embs[row['path1']]
        # else:
        #     audio_1 = decode_audio(os.path.join('./data', row['path1'])).reshape(
        #         (1, -1, 1))  # Load the user enrolled audio
        #     input_1 = get_tf_spectrum(audio_1) if output_type == 'spectrum' else get_tf_filterbanks(
        #         audio_1)  # Extract the acoustic representation
        #     emb_1 = tf.keras.layers.Lambda(lambda emb1: tf.keras.backend.l2_normalize(emb1, 1))(
        #         extractor.predict(input_1))  # Get the speaker embedding
        #     speaker_embs[row['path1']] = emb_1  # Save the current speaker embedding for future usage
        #
        # if row['path2'] in speaker_embs:  # If we already computed the embedding for the second element of the verification pair
        #     emb_2 = speaker_embs[row['path2']]
        # else:
        #     audio_2 = decode_audio(os.path.join('./data', row['path2'])).reshape(
        #         (1, -1, 1))  # Load the master voice audio
        #     if args.playback == 1:
        #         print('> playback and recording simulated successfully')
        #         audio_2 = get_play_n_rec_audio(audio_2, noise_paths, noise_cache,
        #                                        noise_strength='random')  # Simulate playback and recording
        #     input_2 = get_tf_spectrum(audio_2) if output_type == 'spectrum' else get_tf_filterbanks(
        #         audio_2)  # Extract the acoustic representation
        #     emb_2 = tf.keras.layers.Lambda(lambda emb2: tf.keras.backend.l2_normalize(emb2, 1))(
        #         extractor.predict(input_2))  # Get the speaker embedding
        #     speaker_embs[row['path2']] = emb_2  # Save the current master voice speaker embedding for future usage
        #
        # any_scores.append(1 - cosine(emb_1, emb_2))  # Compute the cosine similarity between the two embeddings
        # avg_speaker_embs.append(emb_1)  # Add the current embedding to the list of embeddings of the current user
        # avg_speaker_files.append(
        #     row['path1'])  # Add the current enrolled audio to the list of audio files of the current user
        #
        # if (index + 1) % args.n_templates == 0:  # When we analyze all the enrolled audio file for the current user
        #     print('\r> pair', index + 1, '/', len(df_trial_pairs.index), '-', mv_set, version, mv_csv_file, end='')
        #
        #     # Compute cosine similarity between the averaged embedding and the master voice embedding
        #     avg_scores.append(1 - cosine(np.average(avg_speaker_embs, axis=0), emb_2))
        #     avg_speaker_sets.append((','.join(avg_speaker_files)))
        #     avg_gender.append(row['gender'])
        # 
        #     # Reset the avg 10 embedding list (even if not in use)
        #     avg_speaker_embs, avg_speaker_files = [], []