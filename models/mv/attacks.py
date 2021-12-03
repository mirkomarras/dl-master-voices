import numpy as np
from numpy.core.function_base import _add_docstring
import tensorflow as tf
from helpers import audio
from models import ae  # cloning

from rtvc import rtvc_api


def _nes(input, f, n=10, sigma=0.1, antithetic=True):
    grad = tf.convert_to_tensor(np.zeros_like(input))

    for _ in range(n):
        d = tf.random.normal(input.shape)
        grad += d * tf.reduce_mean(f(input + d * sigma))
        if antithetic:
            grad -= d * tf.reduce_mean(f(input - d * sigma))
        
    grad /= sigma * n * (antithetic + 1) 
    return grad


class Attack(object):
    """
    Implements an attack strategy.
    """

    def setup(self, seed_sample):
        """
        Setup the attack and return: 
        - seed voice sample in the domain needed for the attack
        - attack vector - a vector that defines the attack and that will evolve as a result of the optimization

        Based on both outputs, a call to run(seed_sample, attack_vector) should return a valid attack sample.
        """
        raise NotImplementedError()

    def optimize(self, seed_sample, attack_vector, train_data, settings):
        """
        Optimize the attack vector for a given seed sample and a training population.
        """
        raise NotImplementedError()

    def run(self, seed_sample, attack_vector):
        """
        Generate an attack sample based on the seed sample and an attack vector.
        """
        raise NotImplementedError()


class PGDWaveformDistortion(Attack):

    def __init__(self, siamese_model, playback=False, ir_paths=None, ir_cache=None):
        super().__init__()
        self.siamese_model = siamese_model
        self.playback = playback
        self.ir_paths = ir_paths
        self.ir_cache = ir_cache

    def setup(self, seed_sample):
        attack_vector = np.zeros_like(seed_sample, dtype=np.float32)
        attack_vector = tf.convert_to_tensor(attack_vector, dtype='float32')
        return seed_sample, attack_vector

    def optimize(self, seed_sample, attack_vector, train_data, settings):
        epoch_similarities = []
        for step, batch_data in enumerate(train_data):

            with tf.GradientTape() as tape:

                tape.watch(attack_vector)
                input_mv = self.run(seed_sample[tf.newaxis, ...], attack_vector)

                # Playback simulation
                if self.playback:
                    input_mv = audio.get_play_n_rec_audio(input_mv[..., tf.newaxis], self.ir_paths, self.ir_cache)

                # Convert to spectrogram
                input_mv = audio.get_tf_spectrum(input_mv)
                input_mv = tf.repeat(input_mv, len(batch_data[0]), axis=0)
                loss = self.siamese_model([batch_data[0], input_mv])

                if settings.l2_regularization > 0:
                    loss = loss - tf.reduce_mean(tf.square(attack_vector))

            grads = tape.gradient(loss, attack_vector)

            if settings.gradient == 'pgd':
                grads = tf.sign(grads)
            elif settings.gradient == 'normed':
                grads = grads / (1e-9 + tf.linalg.norm(grads))
            elif settings.gradient is None:
                pass
            else:
                raise ValueError('Unsupported gradient mode!')

            attack_vector += settings.learning_rate * grads
            if settings.max_attack_vector is not None and settings.max_attack_vector > 0:
                attack_vector = tf.clip_by_value(attack_vector, -settings.max_attack_vector, settings.max_attack_vector)

            epoch_similarities.append(tf.reduce_mean(loss).numpy().item())

        return attack_vector, epoch_similarities

    def run(self, seed_sample, attack_vector):
        input_mv = seed_sample + attack_vector
        return input_mv


class PGDSpectrumDistortion(Attack):

    def __init__(self, siamese_model):
        super().__init__()
        self.siamese_model = siamese_model

    def setup(self, seed_sample):
        input_spec = audio.get_tf_spectrum(seed_sample[tf.newaxis, ...])
        attack_vector = np.zeros_like(input_spec, dtype=np.float32)
        attack_vector = tf.convert_to_tensor(attack_vector, dtype='float32')
        return input_spec, attack_vector

    def optimize(self, seed_sample, attack_vector, train_data, settings):
        epoch_similarities = []
        for step, batch_data in enumerate(train_data):

            with tf.GradientTape() as tape:

                attack_vector_repeated = tf.repeat(attack_vector, len(batch_data[0]), axis=0)
                tape.watch(attack_vector_repeated)
                input_mv = seed_sample + attack_vector_repeated

                # input_mv = tf.clip_by_value(input_mv, 0, 10000)
                loss = self.siamese_model([batch_data[0], input_mv])

                if settings.l2_regularization > 0:
                    loss = loss - tf.reduce_mean(tf.square(attack_vector))

            grads = tape.gradient(loss, attack_vector_repeated)

            # Find viable speakers worth pursuing (e.g,. closer than a certain distance)
            # idxs = tf.where(tf.reshape(loss, (-1,)) > settings.min_sim)
            # grads = tf.gather(grads, tf.reshape(idxs, (-1,)))

            if len(grads) > 0:
                grad = tf.reduce_mean(grads, axis=0)

                if settings.gradient == 'pgd':
                    grad = tf.sign(grad)
                elif settings.gradient == 'normed':
                    grad = grad / (1e-9 + tf.linalg.norm(grad))
                elif settings.gradient is None or settings.gradient == 'none':
                    pass
                else:
                    raise ValueError('Unsupported gradient mode!')

                attack_vector += settings.learning_rate * grad
                if settings.max_attack_vector is not None and settings.max_attack_vector > 0:
                    attack_vector = tf.clip_by_value(attack_vector, -settings.max_attack_vector, settings.max_attack_vector)

            epoch_similarities.append(tf.reduce_mean(loss).numpy().item())

        return attack_vector, epoch_similarities

    def run(self, seed_sample, attack_vector):
        return seed_sample + attack_vector


class PGDVariationalAutoencoder(Attack):

    def __init__(self, siamese_model, dataset, version, z_dim):
        super().__init__()
        self.siamese_model = siamese_model
        self.gm = ae.VariationalAutoencoder(dataset, version=version, z_dim=z_dim, patch_size=256)
        self.gm.load()

    def setup(self, seed_sample):
        input_spec = audio.get_tf_spectrum(seed_sample[tf.newaxis, ...], normalized=False)
        m, lv = self.gm.encode(input_spec)
        attack_vector = self.gm.reparameterize(m, lv)
        # attack_vector = np.zeros_like(input_spec, dtype=np.float32)
        # attack_vector = tf.convert_to_tensor(attack_vector, dtype='float32')
        return input_spec, attack_vector

    def optimize(self, seed_sample, attack_vector, train_data, settings):
        epoch_similarities = []
        for step, batch_data in enumerate(train_data):

            with tf.GradientTape() as tape:

                tape.watch(attack_vector)

                input_mv = self.gm.decode(attack_vector)
                input_mv = tf.repeat(input_mv, len(batch_data[0]), axis=0)

                # input_mv = tf.clip_by_value(input_mv, 0, 10000)
                loss = self.siamese_model([batch_data[0], input_mv])

                if settings.l2_regularization > 0:
                    loss = loss - tf.reduce_mean(tf.square(attack_vector))

            grads = tape.gradient(loss, attack_vector)

            # Find viable speakers worth pursuing (e.g,. closer than a certain distance)
            # idxs = tf.where(tf.reshape(loss, (-1,)) > settings.min_sim)
            # grads = tf.gather(grads, tf.reshape(idxs, (-1,)))

            if len(grads) > 0:
                grad = tf.reduce_mean(grads, axis=0)

                if settings.gradient == 'pgd':
                    grad = tf.sign(grad)
                elif settings.gradient == 'normed':
                    grad = grad / (1e-9 + tf.linalg.norm(grad))
                elif settings.gradient is None or settings.gradient == 'none':
                    pass
                else:
                    raise ValueError('Unsupported gradient mode!')

                attack_vector += settings.learning_rate * grad

                if settings.max_attack_vector is not None and settings.max_attack_vector > 0:
                    attack_vector = tf.clip_by_value(attack_vector, -settings.max_attack_vector, settings.max_attack_vector)

            epoch_similarities.append(tf.reduce_mean(loss).numpy().item())

        return attack_vector, epoch_similarities

    def run(self, seed_sample, attack_vector):
        return self.gm.decode(attack_vector)

class NESVoiceCloning(object):

    def __init__(self, siamese_model, text='Hello Google', n=10, sigma=0.01, antithetic=True):
        super().__init__()
        self.siamese_model = siamese_model
        self.n = n
        self.sigma = sigma
        self.antithetic = antithetic
        self.text = text
        rtvc_api.load_models('rtvc')

    def setup(self, seed_sample):
        embedding = rtvc_api.get_embedding(seed_sample)
        seed_sample = self.run(seed_sample, embedding)
        # embedding = cloning.init_embedding(seed_sample)
        return seed_sample, embedding


    def optimize(self, seed_sample, attack_vector, train_data, settings):
        epoch_similarities = []
        for step, batch_data in enumerate(train_data):

            def f(attack_vector):
                input_mv = self.run(seed_sample, attack_vector)
                # if the received sample is too short, pad with zeros
                min_lenght = 16000 * 2.57
                if len(input_mv) < min_lenght:
                    pad_neeed = int(min_lenght - len(input_mv))
                    input_mv = tf.pad(input_mv, [[0, pad_neeed]], 'CONSTANT')
                input_mv = audio.get_tf_spectrum(input_mv[tf.newaxis, ...])
                input_mv = tf.repeat(input_mv, len(batch_data[0]), axis=0)
                loss = self.siamese_model([batch_data[0], input_mv])
                return loss

            loss = f(attack_vector)
            grad = _nes(attack_vector, f, self.n, self.sigma, self.antithetic)
            
            assert grad.numpy().size == 256, 'Gradient shape mismatch'

            if settings.gradient == 'pgd':
                grad = tf.sign(grad)
            elif settings.gradient == 'normed':
                grad = grad / (1e-9 + tf.linalg.norm(grad))
            else:
                raise ValueError('Unsupported gradient mode!')

            attack_vector += settings.learning_rate * grad
            # TODO [Check] Do we need to project back onto the embedding manifold?
            epoch_similarities.append(tf.reduce_mean(loss).numpy().item())

        return attack_vector, epoch_similarities

    def run(self, seed_sample, attack_vector):
        if not isinstance(attack_vector, np.ndarray):
            attack_vector = attack_vector.numpy()
        
        input_mv = rtvc_api.vc(self.text, attack_vector)
        
        # input_mv = cloning.clone_voice(attack_vector, self.text)
        return audio.ensure_length(input_mv, int(2.58 * 16000))


class NESWaveform(Attack):

    def __init__(self, siamese_model, playback=False, ir_paths=None, ir_cache=None):
        super().__init__()
        self.siamese_model = siamese_model
        self.playback = playback
        self.ir_paths = ir_paths
        self.ir_cache = ir_cache
        self.n = 100
        self.sigma = 0.01
        self.antithetic = True

    def setup(self, seed_sample):
        attack_vector = np.zeros_like(seed_sample, dtype=np.float32)
        attack_vector = tf.convert_to_tensor(attack_vector, dtype='float32')
        return seed_sample, attack_vector

    def optimize(self, seed_sample, attack_vector, train_data, settings):
        epoch_similarities = []
        for step, batch_data in enumerate(train_data):

            def f(attack_vector):
                tape.watch(attack_vector)
                input_mv = self.run(seed_sample[tf.newaxis, ...], attack_vector)

                if self.playback:
                    input_mv = audio.get_play_n_rec_audio(input_mv[..., tf.newaxis], self.ir_paths, self.ir_cache)

                # Convert to spectrogram / filterbanks
                if batch_data[0].shape[-1] > 128:
                    input_mv = audio.get_tf_spectrum(input_mv)                
                else:
                    input_mv = audio.get_tf_filterbanks(input_mv[..., tf.newaxis], n_filters=24)

                input_mv = tf.repeat(input_mv, len(batch_data[0]), axis=0)
                loss = self.siamese_model([batch_data[0], input_mv])

                return loss

            with tf.GradientTape() as tape:
                loss = f(attack_vector)
            grads = _nes(attack_vector, f, self.n, self.sigma, self.antithetic)
            # grads = tape.gradient(loss, attack_vector)            

            if settings.gradient == 'pgd':
                grads = tf.sign(grads)
            elif settings.gradient == 'normed':
                grads = grads / (1e-9 + tf.linalg.norm(grads))
            elif settings.gradient is None:
                pass
            else:
                raise ValueError('Unsupported gradient mode!')

            attack_vector += settings.learning_rate * grads
            if settings.max_attack_vector is not None and settings.max_attack_vector > 0:
                attack_vector = tf.clip_by_value(attack_vector, -settings.max_attack_vector, settings.max_attack_vector)

            epoch_similarities.append(tf.reduce_mean(loss).numpy().item())

        return attack_vector, epoch_similarities

    def run(self, seed_sample, attack_vector):
        input_mv = seed_sample + attack_vector
        return input_mv