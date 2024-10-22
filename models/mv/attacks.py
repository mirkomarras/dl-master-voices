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
    Abstract class defining the interface for implementing attacks.
    """

    def setup(self, seed_sample):
        """
        Setup the attack and return: 
        - seed voice sample in the domain needed for the attack
        - attack vector - a vector that defines the attack and that will evolve as a result of the optimization

        Based on both outputs, a call to run(seed_sample, attack_vector) should return a valid attack sample.
        """
        raise NotImplementedError()

    def attack_step(self, seed_sample, attack_vector, population, settings):
        """
        Run one step (pass over the population) of the attack.
        """
        raise NotImplementedError()

    def run(self, seed_sample, attack_vector):
        """
        Generate an attack sample based on the seed sample and an attack vector.
        """
        raise NotImplementedError()


class PGDWaveformDistortion(Attack):

    def __init__(self, siamese_model, playback=False, ir_paths=None, ir_cache=None, impulse_flags=[1,1,1], sgd=True):
        super().__init__()
        self.siamese_model = siamese_model
        self.sgd = sgd
        self.playback = playback
        self.ir_paths = ir_paths
        self.ir_cache = ir_cache
        self.impulse_flags = impulse_flags

    def setup(self, seed_sample):
        attack_vector = np.zeros_like(seed_sample, dtype=np.float32)
        attack_vector = tf.convert_to_tensor(attack_vector, dtype='float32')
        return seed_sample, attack_vector

    def attack_step(self, seed_sample, attack_vector, population, settings):
        epoch_similarities = []
        grads_agg = tf.zeros(attack_vector.shape)        
        # n_batches = len(list(population))

        for step, batch_data in enumerate(population):

            with tf.GradientTape() as tape:

                tape.watch(attack_vector)
                input_mv = self.run(seed_sample, attack_vector)

                # Playback simulation
                if self.playback:
                    input_mv = audio.get_play_n_rec_audio(input_mv[..., tf.newaxis], self.ir_paths, self.ir_cache, impulse_flags=self.impulse_flags)

                # Convert to spectrogram
                input_mv = audio.get_tf_spectrum(input_mv)
                input_mv = tf.repeat(input_mv, len(batch_data[0]), axis=0)
                loss = self.siamese_model([batch_data[0], input_mv])

                if settings.l2_regularization > 0:
                    loss = loss - settings.l2_regularization * tf.reduce_mean(tf.square(attack_vector))

            epoch_similarities.append(tf.reduce_mean(loss).numpy().item())

            grads = tape.gradient(loss, attack_vector)
            grads_agg += grads

            if settings.gradient == 'pgd':
                grads = tf.sign(grads)
            elif settings.gradient == 'normed':
                grads = grads / (1e-9 + tf.linalg.norm(grads))
            elif settings.gradient is None or settings.gradient == 'none':
                pass
            else:
                raise ValueError('Unsupported gradient mode!')

            if not self.sgd:
                continue

            # Update the attack vector after every batch:
            if settings.step_size_override:
                # Override step size if requested
                attack_vector += settings.step_size_override * grads
            elif settings.epsilon:
                # Otherwise, adjust the step size based on the distortion budget and the number of steps
                attack_vector = attack_vector + grads * settings.epsilon / (settings.n_steps)
            else:
                raise ValueError('Unspecified step size!')

            # Optionally clip the attack vector 
            if settings.clip_av:
                attack_vector = tf.clip_by_value(attack_vector, -settings.clip_av, settings.clip_av)            
        
        if not self.sgd:

            # Process the gradient
            if settings.gradient == 'pgd':
                grads_agg = tf.sign(grads_agg)
            elif settings.gradient == 'normed':
                grads_agg = grads_agg / (1e-9 + tf.linalg.norm(grads_agg))
            elif settings.gradient is None or settings.gradient == 'none':
                pass
            else:
                raise ValueError('Unsupported gradient mode!')

            if settings.step_size_override:
                attack_vector += settings.step_size_override * grads_agg
            else:
                attack_vector += settings.epsilon * grads_agg /settings.n_steps

        return attack_vector, tf.reduce_mean(epoch_similarities)

    def run(self, seed_sample, attack_vector):
        input_mv = seed_sample + attack_vector
        input_mv = tf.clip_by_value(input_mv, -1, 1)
        return input_mv


class PGDSpectrumDistortion(Attack):

    def __init__(self, siamese_model):
        super().__init__()
        self.siamese_model = siamese_model

    def setup(self, seed_sample):
        input_spec = audio.get_tf_spectrum(seed_sample)
        attack_vector = np.zeros_like(input_spec, dtype=np.float32)
        attack_vector = tf.convert_to_tensor(attack_vector, dtype='float32')
        return input_spec, attack_vector

    def attack_step(self, seed_sample, attack_vector, population, settings):
        epoch_similarities = []
        # n_batches = len(list(population))

        for step, batch_data in enumerate(population):

            with tf.GradientTape() as tape:

                attack_vector_repeated = tf.repeat(attack_vector, len(batch_data[0]), axis=0)
                tape.watch(attack_vector_repeated)
                input_mv = seed_sample + attack_vector_repeated

                # input_mv = tf.clip_by_value(input_mv, 0, 10000)
                loss = self.siamese_model([batch_data[0], input_mv])

                if settings.l2_regularization > 0:
                    loss = loss - tf.reduce_mean(tf.square(attack_vector))

            grads = tape.gradient(loss, attack_vector_repeated)
            grads = tf.reduce_mean(grads, axis=0)

            if settings.gradient == 'pgd':
                grads = tf.sign(grads)
            elif settings.gradient == 'normed':
                grads = grads / (1e-9 + tf.linalg.norm(grads))
            elif settings.gradient is None or settings.gradient == 'none':
                pass
            else:
                raise ValueError('Unsupported gradient mode!')

            # Update the attack vector after every batch:
            if settings.step_size_override:
                # Override step size if requested
                attack_vector += settings.step_size_override * grads
            elif settings.epsilon:
                # Otherwise, adjust the step size based on the distortion budget and the number of steps
                attack_vector = attack_vector + grads * settings.epsilon / (settings.n_steps)
            else:
                raise ValueError('Unspecified step size!')

            # Optionally clip the attack vector 
            if settings.clip_av:
                attack_vector = tf.clip_by_value(attack_vector, -settings.clip_av, settings.clip_av)

            epoch_similarities.append(tf.reduce_mean(loss).numpy().item())
        
        return attack_vector, tf.reduce_mean(epoch_similarities)

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

    def attack_step(self, seed_sample, attack_vector, target_pop, settings):
        epoch_similarities = []
        for step, batch_data in enumerate(target_pop):

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
        if not isinstance(seed_sample, np.ndarray):
            seed_sample = seed_sample.numpy()
        embedding = rtvc_api.get_embedding(seed_sample.ravel())
        seed_sample = self.run(seed_sample, embedding)
        return seed_sample, embedding

    def attack_step(self, seed_sample, attack_vector, population, settings):
        epoch_similarities = []
        n_batches = len(list(population))

        for step, batch_data in enumerate(population):

            def f(attack_vector):
                input_mv = self.run(seed_sample, attack_vector)
                
                # if the received sample is too short, pad with zeros
                min_lenght = 16000 * 2.57
                if len(input_mv) < min_lenght:
                    pad_neeed = int(min_lenght - len(input_mv))
                    input_mv = tf.pad(input_mv, [[0, pad_neeed]], 'CONSTANT')

                # Convert to spectrogram / filterbanks
                if batch_data[0].shape[2] > 128:
                    input_mv = audio.get_tf_spectrum(input_mv)                
                else:
                    input_mv = audio.get_tf_filterbanks(input_mv[tf.newaxis, ..., tf.newaxis], n_filters=24)

                input_mv = tf.repeat(input_mv, len(batch_data[0]), axis=0)
                loss = self.siamese_model([batch_data[0], input_mv])
                return loss

            loss = f(attack_vector)

            if settings.nes_n is not None and settings.nes_sigma is not None:
                grad = _nes(attack_vector, f, settings.nes_n, settings.nes_sigma, True)
            else:
                grad = _nes(attack_vector, f, self.n, self.sigma, self.antithetic)
            
            assert grad.numpy().size == 256, 'Gradient shape mismatch'

            if settings.gradient == 'pgd':
                grad = tf.sign(grad)
            elif settings.gradient == 'normed':
                grad = grad / (1e-9 + tf.linalg.norm(grad))
            else:
                raise ValueError('Unsupported gradient mode!')

            # Update the attack vector after every batch:
            if settings.step_size_override:
                # Override step size if requested
                attack_vector += settings.step_size_override * grad
            elif settings.epsilon:
                # Otherwise, adjust the step size based on the distortion budget and the number of steps
                attack_vector = attack_vector + grad * settings.epsilon / (settings.n_steps * n_batches)
            else:
                raise ValueError('Unspecified step size!')

            # Optionally clip the attack vector 
            if settings.clip_av:
                attack_vector = tf.clip_by_value(attack_vector, -settings.clip_av, settings.clip_av)

            # TODO [Check] Do we need to project back onto the embedding manifold?
            epoch_similarities.append(tf.reduce_mean(loss).numpy().item())

        return attack_vector, tf.reduce_mean(epoch_similarities)

    def run(self, seed_sample, attack_vector):
        if not isinstance(attack_vector, np.ndarray):
            attack_vector = attack_vector.numpy()
        
        attack_vector = np.clip(attack_vector, 0, 1)
        input_mv = rtvc_api.vc(self.text, attack_vector)
        
        return audio.ensure_length(input_mv, int(2.58 * 16000))


class NESWaveform(Attack):

    def __init__(self, siamese_model, playback=False, ir_paths=None, ir_cache=None, impulse_flags=(1,1,1), n=20, sigma=0.01):
        super().__init__()
        self.siamese_model = siamese_model
        self.playback = playback
        self.ir_paths = ir_paths
        self.ir_cache = ir_cache
        self.impulse_flags = impulse_flags
        self.n = n
        self.sigma = sigma
        self.antithetic = True

    def setup(self, seed_sample):
        attack_vector = np.zeros_like(seed_sample, dtype=np.float32)
        attack_vector = tf.convert_to_tensor(attack_vector, dtype='float32')
        return seed_sample, attack_vector

    def attack_step(self, seed_sample, attack_vector, population, settings):
        epoch_similarities = []
        # n_batches = len(list(population))

        for step, batch_data in enumerate(population):

            def f(attack_vector):
                tape.watch(attack_vector)
                input_mv = self.run(seed_sample, attack_vector)

                if self.playback:
                    input_mv = audio.get_play_n_rec_audio(input_mv[..., tf.newaxis], self.ir_paths, self.ir_cache, impulse_flags=self.impulse_flags)

                # Convert to spectrogram / filterbanks
                if batch_data[0].shape[2] > 128:
                    input_mv = audio.get_tf_spectrum(input_mv)                
                else:
                    input_mv = audio.get_tf_filterbanks(input_mv[..., tf.newaxis], n_filters=24)

                input_mv = tf.repeat(input_mv, len(batch_data[0]), axis=0)
                loss = self.siamese_model([batch_data[0], input_mv])

                if settings.l2_regularization > 0:
                    loss = loss - settings.l2_regularization * tf.reduce_mean(tf.square(attack_vector))

                return loss

            # Compute the loss and the gradient
            with tf.GradientTape() as tape:
                loss = f(attack_vector)

            if settings.nes_n is not None and settings.nes_sigma is not None:
                grads = _nes(attack_vector, f, settings.nes_n, settings.nes_sigma, True)
            else:
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

            # Update the attack vector after every batch:
            if settings.step_size_override:
                # Override step size if requested
                attack_vector += settings.step_size_override * grads
            elif settings.epsilon:
                # Otherwise, adjust the step size based on the distortion budget and the number of steps
                attack_vector = attack_vector + grads * settings.epsilon / (settings.n_steps)
            else:
                raise ValueError('Unspecified step size!')

            # Optionally clip the attack vector 
            if settings.clip_av:
                attack_vector = tf.clip_by_value(attack_vector, -settings.clip_av, settings.clip_av)

            epoch_similarities.append(tf.reduce_mean(loss).numpy().item())

        return attack_vector, tf.reduce_mean(epoch_similarities)

    def run(self, seed_sample, attack_vector):
        input_mv = seed_sample + attack_vector
        input_mv = tf.clip_by_value(input_mv, -1, 1)
        return input_mv
