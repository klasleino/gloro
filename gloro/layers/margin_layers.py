import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.layers import Layer

from gloro.lipschitz_computation import LipschitzComputer
from gloro.utils import batch_set_value
from gloro.utils import get_value
from gloro.utils import set_value


class LipschitzMargin(Layer):
    def __init__(self, epsilon, model, num_iterations=5, norm='l2', **kwargs):
        super().__init__(**kwargs)

        self._epsilon = tf.Variable(
            epsilon, dtype=K.floatx(), name='epsilon', trainable=False)
        self._num_iterations = tf.Variable(
            num_iterations, name='num_iterations', trainable=False)

        # Check norm compatibility.
        if norm != 'l2':
            # For now we only support the l2 norm.
            raise ValueError(f'{norm} norm not supported')

        self._norm = norm

        self._layer_computers = LipschitzComputer.for_model(
            model, self._num_iterations)
        self._W = model.layers[-1].kernel

        self._lc_frozen = tf.Variable(False, name='lc_frozen', trainable=False)
        self._hardcoded_lc = tf.Variable(
            1., name='hardcoded_lc', trainable=False)

    @property
    def epsilon(self):
        return get_value(self._epsilon)

    @epsilon.setter
    def epsilon(self, new_epsilon):
        set_value(self._epsilon, new_epsilon)

    @property
    def num_iterations(self):
        return get_value(self._num_iterations)

    @num_iterations.setter
    def num_iterations(self, new_num_iterations):
        set_value(self._num_iterations, new_num_iterations)

    @property
    def norm(self):
        return self._norm

    @property
    def frozen(self):
        return get_value(self._lc_frozen)

    def freeze(self):
        set_value(self._hardcoded_lc, self._lc())
        set_value(self._lc_frozen, True)

    
    def build(self, input_shape):

        self._lc = lambda: (
            self._hardcoded_lc if self._lc_frozen else
            LipschitzComputer.global_lipschitz_bound(self._layer_computers))

        # This is the L2 norm of the Lipschitz constant of the penultimate layer
        # of the network multiplied by the weights of the penulimate layer.
        self._kW = lambda: self._lc() * self._W

    def _get_yj_j_Kij(self, y):
        """
        `j` is the index of the predicted class, and `y_j` is the corresponding
        max logit value. `y_j` has shape (None, 1).

        `K_ij` is the Lipschitz constant for the function $f_j(x) - f_i(x)$ for
        i != j. That is, the ith entry of `K_ij` contains the Lipschitz constant
        of $f_j(x) - f_i(x)$, where `j` is as defined above. `K_ij` has shape
        (None, m).
        """
        kW = self._kW()

        j = tf.argmax(y, axis=1)
        y_j = tf.reduce_max(y, axis=1, keepdims=True)

        # Get the weight column of the predicted class.
        kW_j = tf.gather(tf.transpose(kW), j)

        # Get weights that predict the value y_j - y_i for all i != j.
        kW_ij = kW_j[:,:,None] - kW[None]
        
        # We do this instead of `tf.linalg.norm(W_d)` because of an apparent bug
        # in `tf.linalg.norm` that leads to NaN values.
        K_ij = tf.sqrt(tf.reduce_sum(kW_ij * kW_ij, axis=1))

        return y_j, j, K_ij
    
    def lipschitz_constant(self, y):

        y_j, j, K_ij = self._get_yj_j_Kij(y)

        return tf.where(
            tf.equal(y, y_j), 
            tf.zeros_like(K_ij) - 1., 
            K_ij)

    def certified_radius(self, y):

        y_j, j, K_ij = self._get_yj_j_Kij(y)

        # y.shape: (None, m); y_j.shape: (None, 1)

        margins = y_j - y # shape: (None, m)

        # This is a certified lower bound on the distance from the point to the
        # boundary with class i, where i != j.
        radius_i = tf.where(
            tf.equal(y, y_j),
            np.infty + tf.zeros_like(y),
            margins / K_ij)

        return tf.reduce_min(radius_i, axis=1)

    def call(self, y):
        
        y_j, j, K_ij = self._get_yj_j_Kij(y)

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
        verbose=False,
    ):
        if self._lc_frozen:
            return self

        batch_set_value([
            (
                layer.iterate, 
                tf.random.truncated_normal(layer.iterate.shape))
            for layer in self._layer_computers
            if hasattr(layer, 'iterate')
        ])

        if converge:
            if batch_size:
                old_num_iterations = self.num_iterations
                self.num_iterations = batch_size

            i = 0
            k_prev = None
            k = self._lc()

            while i < max_tries and (
                k_prev is None or abs(k - k_prev) > convergence_threshold
            ):
                k_prev = k
                k = self._lc()
                i += 1

            if verbose:
                print(
                    f'power method converged in {(i+1)*self.num_iterations} '
                    f'iterations to {get_value(k):.3f}')

            if batch_size:
                self.num_iterations = old_num_iterations

        return self

    def update_iterates(self):
        if self._lc_frozen:
            return self

        self._lc()

        return self
