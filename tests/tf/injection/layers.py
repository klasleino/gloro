import tensorflow as tf

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input


def with_shape(*shape):
    def decorator(fn):
        new_fn = lambda self: fn(self, shape)
        new_fn.input_shape = shape

        return new_fn

    return decorator

class Layers(object):
    @with_shape(32,)
    def orthonormal_dense(self, input_shape):
        layer = Dense(32, kernel_initializer='orthogonal')
        layer(Input(input_shape))
        return layer

    @with_shape(32,)
    def random_dense(self, input_shape):
        layer =  Dense(
            32,
            kernel_initializer='glorot_uniform',
            bias_initializer='random_normal',
        )
        layer(Input(input_shape))
        return layer

    @with_shape(32,32,16)
    def random_conv(self, input_shape):
        layer = Conv2D(
            16,
            3,
            padding='same',
            kernel_initializer='glorot_uniform',
            bias_initializer='random_normal',
        )
        layer(Input(input_shape))
        return layer

    @with_shape(32,32,16)
    def strided_conv(self, input_shape):
        layer = Conv2D(
            16,
            4,
            strides=2,
            padding='same',
            kernel_initializer='glorot_uniform',
            bias_initializer='random_normal',
        )
        layer(Input(input_shape))
        return layer
