import tensorflow as tf

from gloro import GloroNet
from gloro.layers import Conv2D
from gloro.layers import Dense
from gloro.layers import Flatten
from gloro.layers import Input
from gloro.layers import InvertibleDownsampling
from gloro.layers import MinMax


def with_shape(*shape):
    def decorator(fn):
        new_fn = lambda self, **kwargs: fn(self, shape, **kwargs)
        new_fn.input_shape = shape

        return new_fn

    return decorator

class Models(object):
    @with_shape(32,)
    def orthonormal_dense(self, input_shape, lc_strategy='eigh'):
        x = Input(input_shape)
        z = Dense(
            32,
            kernel_initializer='orthogonal',
            lc_strategy=lc_strategy,
        )(x)
        z = MinMax()(z)
        z = Dense(
            32,
            kernel_initializer='orthogonal',
            lc_strategy=lc_strategy,
        )(z)
        z = MinMax()(z)
        y = Dense(
            4,
            kernel_initializer='orthogonal',
            lc_strategy=lc_strategy,
        )(z)

        return GloroNet(x, y, 1.)

    @with_shape(32,32,16)
    def random_conv(self, input_shape, lc_strategy='power'):
        x = Input(input_shape)
        z = Conv2D(
            16,
            3,
            padding='same',
            kernel_initializer='glorot_uniform',
            bias_initializer='random_normal',
            lc_strategy=lc_strategy,
        )(x)
        z = MinMax()(z)
        z = InvertibleDownsampling(2)(z)
        z = Conv2D(
            16,
            3,
            padding='same',
            strides=2,
            kernel_initializer='glorot_uniform',
            bias_initializer='random_normal',
            lc_strategy=lc_strategy,
        )(z)
        z = MinMax()(z)
        z = Flatten()(z)
        z = Dense(
            64,
            kernel_initializer='glorot_uniform',
            bias_initializer='random_normal',
            lc_strategy=lc_strategy,
        )(z)
        z = MinMax()(z)
        y = Dense(
            4,
            kernel_initializer='glorot_uniform',
            bias_initializer='random_normal',
            lc_strategy=lc_strategy,
        )(z)

        return GloroNet(x, y, 1.)
