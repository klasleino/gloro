"""
This file contains `GloroLayer`s not found in the standard keras library.
"""
import tensorflow as tf

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Layer

from gloro.layers.base import GloroLayer
from gloro.utils import get_value
from gloro.utils import set_value


class MinMax(GloroLayer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._flat_op = Flatten()

    def call(self, x):
        x_flat = self._flat_op(x)
        x_shape = tf.shape(x_flat)

        grouped_x = tf.reshape(
            x_flat,
            tf.concat([x_shape[:-1], (-1, 2)], -1))

        min_x = tf.reduce_min(grouped_x, axis=-1, keepdims=True)
        max_x = tf.reduce_max(grouped_x, axis=-1, keepdims=True)

        sorted_x = tf.reshape(
            tf.concat([min_x, max_x], axis=-1),
            tf.shape(x))

        return sorted_x

    def lipschitz(self):
        return 1.


class InvertibleDownsampling(GloroLayer):
    def __init__(self, pool_size=2, **kwargs):
        super().__init__(**kwargs)

        self._pool_size = pool_size

    @property
    def pool_size(self):
        if isinstance(self._pool_size, (tuple, list)):
            return self._pool_size

        return (self._pool_size, self._pool_size)

    def call(self, inputs, **kwargs):
        return tf.concat(
            [
                # Expects channels-last format.
                inputs[:, i :: self.pool_size[0], j :: self.pool_size[1], :]
                for i in range(self.pool_size[0])
                for j in range(self.pool_size[1])
            ],
            axis=-1)

    def lipschitz(self):
        return 1.

    def get_config(self):
        config = {
            'pool_size': self._pool_size,
        }
        base_config = super().get_config()

        return dict(list(base_config.items()) + list(config.items()))


class Scaling(GloroLayer):
    def __init__(self, init=1., **kwargs):
        super().__init__(**kwargs)

        self._weight = tf.Variable(init, name='alpha', trainable=True)

    @property
    def alpha(self):
        return get_value(self._weight)

    @alpha.setter
    def alpha(self, new_w):
        set_value(self._weight, new_w)

    def call(self, x):
        return self._weight * x

    def lipschitz(self):
        return self._weight

    def get_config(self):
        config = {
            'init': self.alpha,
        }
        base_config = super().get_config()

        return dict(list(base_config.items()) + list(config.items()))
