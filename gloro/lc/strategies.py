import numpy as np
import tensorflow as tf

from abc import abstractmethod
from tensorflow.keras.layers import AveragePooling2D as KerasAveragePooling2D
from tensorflow.keras.layers import Conv2D as KerasConv2D
from tensorflow.keras.layers import Dense as KerasDense
from tensorflow.keras.layers import Flatten as KerasFlatten
from tensorflow.keras.layers import InputLayer as KerasInput
from tensorflow.keras.layers import MaxPooling2D as KerasMaxPooling2D
from tensorflow.keras.layers import ReLU as KerasReLU

import gloro

from gloro.layers import GloroLayer
from gloro.utils import get_value
from gloro.utils import l2_normalize
from gloro.utils import set_value


class LipschitzComputationStrategy(object):

    @abstractmethod
    def build(W, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def compute(self):
        raise NotImplementedError

    def __call__(self):
        return self.compute()

    @staticmethod
    def get(strategy):
        if isinstance(strategy, LipschitzComputationStrategy):
            return strategy

        if strategy == 'power':
            return PowerMethod()

        elif strategy == 'eigh':
            return Eigh()

        else:
            raise ValueError(
                f'unknown Lipschitz computation strategy: {strategy}'
            )

    @staticmethod
    def for_layer(layer):
        if isinstance(layer, GloroLayer):
            return layer.lipschitz

        # Also support Lipschitz computation on a set of native Keras layers for
        # maximum compatibility. For layers with multiple possible Lipschitz
        # computation strategies, this uses the default, so the `GloroLayer`
        # version should be used if any other strategy is desired.

        if isinstance(layer, KerasDense):
            return PowerMethod().build(layer.kernel)

        elif isinstance(layer, KerasConv2D):
            return PowerMethod().build(
                layer.kernel,
                input_shape=layer.input_shape,
                strides=layer.strides,
                padding=layer.padding.upper(),
            )

        elif isinstance(layer, KerasAveragePooling2D):
            lc = PowerMethod(100).build(
                tf.eye(layer.input.shape[-1])[None,None] * (
                    tf.ones(layer.pool_size)[:,:,None,None]) / (
                        layer.pool_size[0] * layer.pool_size[1]),
                input_shape=layer.input_shape,
                strides=layer.strides,
                padding=layer.padding.upper(),
            ).compute()

            return lambda: lc

        elif isinstance(
            layer, (KerasReLU, KerasMaxPooling2D, KerasInput, KerasFlatten)
        ):
            return lambda: 1.
        
        print(
            f'WARNING: No Lipschitz computation strategy is implemented for '
            f'{type(layer)}. Using 1.0 as the default Lipschitz constant, but '
            f'this may lead to incorrect certification if this is not a valid '
            f'upper bound.'
        )
        return lambda: 1.


class PowerMethod(LipschitzComputationStrategy):
    def __init__(self, iterations=10, convergence_threshold=1e-5):
        self._iterations = tf.Variable(
            iterations,
            dtype='int32',
            trainable=False,
        )
        self._convergence_threshold = tf.Variable(
            convergence_threshold,
            dtype='float32',
            trainable=False,
        )

    @property
    def iterations(self):
        return get_value(self._iterations)

    @iterations.setter
    def iterations(self, new_value):
        set_value(self._iterations, new_value)

    @property
    def convergence_threshold(self):
        return get_value(self._convergence_threshold)

    @convergence_threshold.setter
    def convergence_threshold(self, new_value):
        set_value(self._convergence_threshold, new_value)

    @property
    def iterate(self):
        return self._power_iterate

    def build(self, W, input_shape=None, strides=None, padding=None, **kwargs):
        if len(W.shape) == 2:
            # This handles the case for a dense layer.

            # This is the power iterate, which represents the current estimation
            # of the leading eigenvector.
            self._power_iterate = tf.Variable(
                tf.random.truncated_normal((W.shape[1], 1)),
                dtype='float32',
                trainable=False,
            )
            self._forward = lambda x: W @ x
            self._transpose = lambda Wx: tf.transpose(W) @ Wx
            self._normalize = lambda x: l2_normalize(x, axis=0)

        elif len(W.shape) == 4:
            # This handles the case for a conv layer.

            if input_shape is None:
                # Fail.
                pass

            # This is the power iterate, which represents the current estimation
            # of the leading eigenvector.
            iterate_shape = (1, *input_shape[1:])
            self._power_iterate = tf.Variable(
                tf.random.truncated_normal(iterate_shape),
                dtype='float32',
                trainable=False,
            )

            self._forward = lambda x: tf.nn.conv2d(
                x, W,
                strides=strides,
                padding=padding,
            )
            self._transpose = lambda Wx: tf.nn.conv2d_transpose(
                Wx, W, iterate_shape,
                strides=strides,
                padding=padding,
            )
            self._normalize = lambda x: l2_normalize(x, axis=(1,2,3))

        else:
            raise ValueError('PowerMethod only supports 2D or 4D kernels')

        return self

    def compute(self):
        def body(i, x, diff):
            x_orig = x

            Wx = self._forward(x)
            x = self._transpose(Wx)
            x = self._normalize(x)

            diff = tf.reduce_sum((x - x_orig)**2.)

            return i + 1, tf.stop_gradient(x), diff

        x = self._normalize(self.iterate)

        _, x, _ = tf.while_loop(
            lambda i, _, diff: tf.logical_and(
                i < self._iterations, diff > self._convergence_threshold),
            body,
            [tf.constant(0), x, tf.constant(np.inf)],
        )

        # Update the power iterate.
        self.iterate.assign(x)

        return tf.sqrt(
            tf.reduce_sum((self._forward(x))**2.) / (
                tf.reduce_sum(x**2.) + gloro.constants.EPS
            )
        )


class Eigh(LipschitzComputationStrategy):

    def build(self, W, **kwargs):
        if len(W.shape) != 2:
            raise ValueError('Eigh only supports 2D kernels')

        self._A = lambda: tf.transpose(W) @ W

        return self

    def compute(self):
        return tf.sqrt(tf.reduce_max(tf.linalg.eigh(self._A())[0]))
