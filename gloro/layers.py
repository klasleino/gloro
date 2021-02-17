import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.layers import Layer

from gloro.utils import global_lipschitz_bound_spectral_power
from gloro.utils import np_truncated_normal


class LipschitzMargin(Layer):
    def __init__(
            self, 
            epsilon, 
            model=None,
            num_iterations=5, 
            norm='l2', 
            maintain_state=True,
            hardcoded_kW=None, 
            **kwargs):

        super().__init__(**kwargs)

        self._epsilon = tf.Variable(epsilon, trainable=False)
        self._num_iterations = tf.Variable(num_iterations, trainable=False)

        self._norm = norm
        self._maintain_state = maintain_state if hardcoded_kW is None else False

        if model is None and hardcoded_kW is None:
            raise ValueError('must provide either `model` or `hardcoded_kW`')

        self._model = model
        self._hardcoded_kW = hardcoded_kW

    @property
    def epsilon(self):
        return K.get_value(self._epsilon)

    @epsilon.setter
    def epsilon(self, new_epsilon):
        K.set_value(self._epsilon, new_epsilon)

    @property
    def num_iterations(self):
        return K.get_value(self._num_iterations)

    @num_iterations.setter
    def num_iterations(self, new_num_iterations):
        K.set_value(self._num_iterations, new_num_iterations)

    @property
    def norm(self):
        return self._norm

    @property
    def maintain_state(self):
        return self._maintain_state

    @property
    def non_trainable_weights(self):
        return [self._epsilon, self._num_iterations]

    def call(self, y):
        if self._hardcoded_kW is None:
            k = global_lipschitz_bound_spectral_power(
                self._model, 
                self._num_iterations, 
                self._norm, 
                maintain_state=self._maintain_state)

            if self._maintain_state:
                k, iterates, updates = k

                self._iterates = iterates

                if not tf.executing_eagerly():
                    self._update_fn = K.function([], [k], updates=updates)

                else:
                    self._update_fn = K.function([], [k])

            # This is the L2 norm of the Lipschitz constant of the penultimate 
            # layer of the network multiplied by the weights of the penulimate 
            # layer.
            kW = k * self._model.layers[-1].kernel

        else:
            kW = self._hardcoded_kW

        self._kW = kW

        j = tf.argmax(y, axis=1)
        y_j = tf.reduce_max(y, axis=1, keepdims=True)
        where_not_j = tf.not_equal(y, y_j)

        # Get the weight column of the predicted class.
        kW_j = tf.gather(tf.transpose(kW), j)

        # Get weights that predict the value y_j - y_i for all i != j.
        kW_ij = kW_j[:,:,None] - kW[None]
        
        # We do this instead of `tf.linalg.norm(W_d)` because of an apparent bug
        # in `tf.linalg.norm` that leads to NaN values.
        K_ij = tf.sqrt(tf.reduce_sum(kW_ij * kW_ij, axis=1) + 1.e-10)

        # Save the margin Lipschitz constant for future reference. Put -1 in the 
        # position of j to signify that this Lipschitz constant is not defined
        # in this position.
        self._lipschitz_constant_tensor = tf.where(
            tf.equal(y, y_j), 
            tf.zeros_like(K_ij) - 1., 
            K_ij)

        y_bot_i = y + self._epsilon * K_ij

        # `y_bot_i` will be zero at the position of class j. However, we don't 
        # want to consider this class, so we replace the zero with negative
        # infinity so that when we find the maximum component for `y_bot_i` we 
        # don't get zero as a result of all of the components we care aobut 
        # being negative.
        y_bot_i = tf.where(
            tf.equal(y, y_j), 
            -np.infty + tf.zeros_like(y_bot_i), 
            y_bot_i)

        y_bot = tf.reduce_max(y_bot_i, axis=1, keepdims=True)

        return tf.concat([y, y_bot], axis=1)

    def refresh_iterates(
            self, 
            converge=True, 
            convergence_threshold=1e-4, 
            max_tries=1000, 
            batch_size=None,
            verbose=False):

        if not self._maintain_state:
            return self

        K.batch_set_value([
            (iterate, np_truncated_normal(K.int_shape(iterate)))
            for iterate in self._iterates
        ])

        if converge:
            if batch_size:
                old_num_iterations = self.num_iterations
                self.num_iterations = batch_size

            i = 0
            k_prev = None
            k = self._update_fn([])[0]

            while i < max_tries and (
                    k_prev is None or abs(k - k_prev) > convergence_threshold):

                k_prev = k
                k = self._update_fn([])[0]
                i += 1

            if batch_size:
                self.num_iterations = old_num_iterations

            if verbose:
                print(f'power method converged in {i+1} iterations')

        return self

    def update_iterates(self):
        if not self._maintain_state:
            return self

        self._update_fn([])

        return self

    def get_config(self):
        return {
            'epsilon': self.epsilon, 
            'num_iterations': self.num_iterations,
            'norm': self.norm,
            'name': self.name,

            # When saving, we freeze the Lipschitz constant so we don't have to
            # keep track of the model.
            'model': None,
            'hardcoded_kW': K.get_value(self._kW),
            'maintain_state': False,
        }
