"""
This file contains various ResNet blocks implemented for GloRo.
"""
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.layers import Conv2D as KerasConv2D

from gloro.layers.base import GloroLayer
from gloro.lc import LipschitzComputationStrategy
from gloro.utils import get_value
from gloro.utils import set_value


class LiResNetBlock(GloroLayer):

    def __init__(
        self,
        kernel_size,
        use_bias=True,
        use_affine=True,
        residual_scale=1.,
        lc_strategy=None,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._kernel_size = (
            (kernel_size, kernel_size) if isinstance(kernel_size, int) else
            kernel_size
        )
        self._use_bias = use_bias

        self._use_affine = use_affine
        self._residual_scale = self._epsilon = tf.Variable(
            residual_scale, dtype=K.floatx(), name='scale', trainable=False
        )
        self._lc_strategy = LipschitzComputationStrategy.get(
            lc_strategy or 'power'
        )

        self._other_conv_kwargs = {
            'use_bias': use_bias,
            'kernel_initializer': kernel_initializer,
            'bias_initializer': bias_initializer,
            'kernel_regularizer': kernel_regularizer,
            'bias_regularizer': bias_regularizer,
            'activity_regularizer': activity_regularizer,
            'kernel_constraint': kernel_constraint,
            'bias_constraint': bias_constraint,
        }


    @property
    def kernel_size(self):
        return self._kernel_size

    @property
    def use_bias(self):
        return self._use_bias

    @property
    def use_affine(self):
        return self._use_affine

    @property
    def residual_scale(self):
        return self._residual_scale

    @residual_scale.setter
    def residual_scale(self, new_value):
        set_value(self._residual_scale, new_value)

    @property
    def lc_strategy(self):
        return self._lc_strategy

    @property
    def kernel(self):
        return (
            self.residual_scale * self.affine * self.conv_kernel +
            self._identity_kernel
        )

    def build(self, input_shape):
        super().build(input_shape)

        # Since we use an identity skip connection, the number of output
        # channels has to be the same as the number of input channels. Here we
        # are assuming channels-last format.
        channels = input_shape[-1]

        kernel_shape = (*self.kernel_size, channels, channels)

        self.conv_kernel = self.add_weight(
            name='conv_kernel',
            shape=kernel_shape,
            initializer=self._other_conv_kwargs['kernel_initializer'],
            regularizer=self._other_conv_kwargs['kernel_regularizer'],
            constraint=self._other_conv_kwargs['kernel_constraint'],
            trainable=True,
            dtype='float32',
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(channels,),
                initializer=self._other_conv_kwargs['bias_initializer'],
                regularizer=self._other_conv_kwargs['bias_regularizer'],
                constraint=self._other_conv_kwargs['bias_constraint'],
                trainable=True,
                dtype='float32',
            )

        if self.use_affine:
            self.affine = self.add_weight(
                name='affine',
                shape=(1, 1, 1, channels),
                initializer='ones',
                trainable=True,
                dtype='float32',
            )
        else:
            self.affine = tf.constant(1., dtype='float32')

        # For input channel i, and output channel j, whenever i = j, we want a
        # filter with exactly one 1 in the center, and 0 everywhere else. The 1
        # is placed in the center (or defaulting to the top left if the kernel
        # size is even) to account for the padding that must be applied to keep
        # the input and output channels the same size.
        idendity_kernel_np = np.zeros(kernel_shape)
        idendity_kernel_np[
            (self.kernel_size[0] - 1) // 2, (self.kernel_size[1] - 1) // 2
        ] = np.eye(channels)

        self._identity_kernel = tf.constant(idendity_kernel_np, dtype='float32')

        self.lc_strategy.build(
            self.kernel,
            input_shape=input_shape,
            strides=1,
            padding='SAME',
        )

    def call(self, x, **kwargs):
        z = tf.nn.conv2d(
            x, self.kernel,
            strides=1,
            padding='SAME',
            name=self.__class__.__name__
        )
        if self.use_bias:
            z = tf.nn.bias_add(z, self.bias)

        if not tf.executing_eagerly():
            # Infer the static output shape.
            z.set_shape(x.shape)

        return z

    def lipschitz(self):
        return self.lc_strategy.compute()

    def get_config(self):
        config = {
            'kernel_size': self.kernel_size,
            'use_bias': self.use_bias,
            'use_affine': self.use_affine,
            'residual_scale': float(self.residual_scale.numpy()),
            'lc_strategy': self.lc_strategy,
            **self._other_conv_kwargs,
        }
        base_config = super().get_config()

        return dict(list(base_config.items()) + list(config.items()))
