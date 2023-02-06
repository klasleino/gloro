import numpy as np
import tensorflow as tf

from gloro.layers import AveragePooling2D
from gloro.layers import Conv2D
from gloro.layers import Dense
from gloro.layers import Input
from gloro.layers import LiResNetBlock
from gloro.layers import Scaling


def with_shape(*shape):
    def decorator(fn):
        new_fn = lambda self, **kwargs: fn(self, shape, **kwargs)
        new_fn.input_shape = shape

        return new_fn

    return decorator

class Layers(object):
    @with_shape(32,)
    def orthonormal_dense(self, input_shape, lc_strategy='power'):
        layer = Dense(
            32,
            kernel_initializer='orthogonal',
            lc_strategy=lc_strategy,
        )
        layer(Input(input_shape))
        return layer

    @with_shape(32,)
    def random_dense(self, input_shape, lc_strategy='power'):
        layer =  Dense(
            32,
            kernel_initializer='glorot_uniform',
            bias_initializer='random_normal',
            lc_strategy=lc_strategy,
        )
        layer(Input(input_shape))
        return layer

    @with_shape(32,32,16)
    def random_conv(self, input_shape, lc_strategy='power'):
        layer = Conv2D(
            16,
            3,
            padding='same',
            kernel_initializer='glorot_uniform',
            bias_initializer='random_normal',
            lc_strategy=lc_strategy,
        )
        layer(Input(input_shape))
        return layer

    @with_shape(32,32,16)
    def strided_conv(self, input_shape, lc_strategy='power'):
        layer = Conv2D(
            16,
            4,
            strides=2,
            padding='same',
            kernel_initializer='glorot_uniform',
            bias_initializer='random_normal',
            lc_strategy=lc_strategy,
        )
        layer(Input(input_shape))
        return layer

    @with_shape(32,32,16)
    def average_pooling(self, input_shape, pool_size=2):
        layer = AveragePooling2D(pool_size)
        layer(Input(input_shape))
        return layer

    @with_shape(32,)
    def scaling(self, input_shape, scale=4.):
        layer = Scaling(scale)
        layer(Input(input_shape))
        return layer

    @with_shape(32,32,16)
    def liresnet_block_no_affine(self, input_shape, lc_strategy='power'):
        layer = LiResNetBlock(
            3,
            use_affine=False,
            residual_scale=2.2,
            kernel_initializer='glorot_uniform',
            bias_initializer='random_normal',
            lc_strategy=lc_strategy,
        )
        layer(Input(input_shape))
        return layer

    @with_shape(32,32,16)
    def liresnet_block_with_affine(self, input_shape, lc_strategy='power'):
        layer = LiResNetBlock(
            3,
            use_affine=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='random_normal',
            lc_strategy=lc_strategy,
        )
        layer(Input(input_shape))

        Ws = layer.get_weights()
        layer.set_weights([
            *Ws[:2],
            4. * np.random.rand(*Ws[2].shape) - 2.,
            Ws[3]
        ])

        return layer

    @with_shape(32,32,16)
    def liresnet_block_identity(
        self, input_shape, kernel_size=3, lc_strategy='power'
    ):
        layer = LiResNetBlock(
            kernel_size,
            kernel_initializer='zeros',
            bias_initializer='zeros',
            lc_strategy=lc_strategy,
        )
        layer(Input(input_shape))
        return layer
