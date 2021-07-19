import tensorflow as tf

from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Layer

import gloro

from gloro.utils import get_value
from gloro.utils import set_value


class MinMax(Layer):
    def __init__(self, absolute_value=False, **kwargs):
        super().__init__(**kwargs)

        self._do_abs = absolute_value
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

        if self._do_abs:
            return tf.math.abs(sorted_x)
        else:
            return sorted_x

    def get_config(self):
        config = {
            'absolute_value': self._do_abs,
        }
        base_config = super().get_config()

        return dict(list(base_config.items()) + list(config.items()))


class InvertibleDownsampling(Layer):
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
                # Expects channels last format.
                inputs[:, i :: self.pool_size[0], j :: self.pool_size[1], :] 
                for i in range(self.pool_size[0])
                for j in range(self.pool_size[1])
            ],
            axis=-1)

    def get_config(self):
        config = {
            'pool_size': self._pool_size,
        }
        base_config = super().get_config()

        return dict(list(base_config.items()) + list(config.items()))


class Scaling(Layer):
    def __init__(self, init=1., **kwargs):
        super().__init__(**kwargs)

        self._weight = tf.Variable(init, name='w', trainable=True)

    @property
    def w(self):
        return get_value(self._weight)

    @w.setter
    def w(self, new_w):
        set_value(self._weight, new_w)
        
    def call(self, x):
        return self._weight * x

    def get_config(self):
        config = {
            'init': self.w,
        }
        base_config = super().get_config()

        return dict(list(base_config.items()) + list(config.items()))


class Bias(Layer):
    def __init__(self, init=0., **kwargs):
        super().__init__(**kwargs)

        self._bias = tf.Variable(init, name='b', trainable=True)

    @property
    def b(self):
        return get_value(self._bias)

    @b.setter
    def b(self, new_b):
        set_value(self._bias, new_b)
        
    def call(self, x):
        return self._bias + x

    def get_config(self):
        config = {
            'init': self.b,
        }
        base_config = super().get_config()

        return dict(list(base_config.items()) + list(config.items()))


class ResnetBlock(object):

    _global_counter = 0

    identifier = 'ResnetBlock'
    join_identifier = 'join'
    skip_identifier = 'skip'

    def __init__(
        self, 
        filters, 
        kernel_sizes=(1,3,1),
        stride1=2,
        depth=None, 
        activation='relu',
        identity_skip=False,
        use_invertible_downsample=False,
        use_fixup_weight_and_bias=False,
        kernel_initializer='orthogonal',
    ):
        n_filters = len(filters) if isinstance(filters, (list, tuple)) else None
        n_kernels = (
            len(kernel_sizes) if isinstance(kernel_sizes, (list, tuple)) else 
            None)

        if (n_filters is not None and n_kernels is not None and 
                n_filters != n_kernels):

            raise ValueError(
                f'if `filters` and `kernel_sizes` are provided as lists, their '
                f'lengths must match, but got {n_filters} and {n_kernels}')

        if depth is None:
            if n_filters is None and n_kernels is None:
                raise ValueError(
                    'must either provide `depth` or give `filters`/'
                    '`kernel_sizes` as a list or tuple')

            depth = n_filters if n_filters is not None else n_kernels

        else:
            if n_filters is not None and n_filters != depth:
                raise ValueError(
                    'length of `filters` must match `depth` (if both given)')

            if n_kernels is not None and n_kernels != depth:
                raise ValueError(
                    'length of `kernel_sizes` must match `depth` (if both '
                    'given)')
        
        if not(stride1 in [None, 1]) and identity_skip:
            raise ValueError(
                'stride1 must be `None` or 1 when using `identity_skip`')
        
        self._filters = (
            [filters for _ in range(depth)] if n_filters is None else 
            filters)
        self._kernel_sizes = (
            [kernel_sizes for _ in range(depth)] if n_kernels is None else 
            kernel_sizes) 
        self._depth = depth
        self._stride1 = stride1

        self._identity_skip = identity_skip
        self._use_invertible_downsample = use_invertible_downsample
        self._use_fixup_weight_and_bias = use_fixup_weight_and_bias
        
        self._initializer = kernel_initializer

        if activation == 'minmax':
            self._activation = lambda name: MinMax(name=name)
        elif isinstance(activation, str):
            self._activation = lambda name: Activation(activation, name=name)
        else:
            self._activation = lambda name: activation(name=name)

        self._block_name = (
            f'{ResnetBlock.identifier}{ResnetBlock._global_counter}')
        ResnetBlock._global_counter += 1

    def __call__(self, input_tensor):
        i = 0
        out = input_tensor
        for filters_i, kernel_size_i in zip(self._filters, self._kernel_sizes):

            if self._use_fixup_weight_and_bias:
                next_layer = Bias(name=f'{self._block_name}_bias_pre{i}')
                next_layer._gloro_branch = gloro.constants.RESIDUAL_BRANCH

                out = next_layer(out)

            next_layer = Conv2D(
                filters_i, 
                kernel_size_i,
                strides=(
                    (self._stride1, self._stride1) 
                    if (i == 1 and (not self._use_invertible_downsample)) else 
                    (1, 1)),
                padding='same',
                use_bias=not self._use_fixup_weight_and_bias,
                kernel_initializer=self._initializer,
                name=f'{self._block_name}_conv{i}')
            next_layer._gloro_branch = gloro.constants.RESIDUAL_BRANCH

            out = next_layer(out)

            if i < self._depth - 1:
                if self._use_fixup_weight_and_bias:
                    next_layer = Bias(name=f'{self._block_name}_bias_post{i}')
                    next_layer._gloro_branch = gloro.constants.RESIDUAL_BRANCH

                    out = next_layer(out)

                next_layer = self._activation(
                    f'{self._block_name}_activation{i}')
                next_layer._gloro_branch = gloro.constants.RESIDUAL_BRANCH

                out = next_layer(out)

            elif self._use_fixup_weight_and_bias:
                next_layer = Scaling(name=f'{self._block_name}_scaling')
                next_layer._gloro_branch = gloro.constants.RESIDUAL_BRANCH

                out = next_layer(out)
            
            if i == 1 and self._stride1 > 1 and self._use_invertible_downsample:
                next_layer = InvertibleDownsampling(
                    self._stride1, 
                    name=f'{self._block_name}_downsampling')
                next_layer._gloro_branch = gloro.constants.RESIDUAL_BRANCH

                out = next_layer(out)

            i += 1

        if self._identity_skip:
            next_layer = Add(
                name=f'{self._block_name}_{ResnetBlock.join_identifier}')
            next_layer._gloro_branch = gloro.constants.MAIN_BRANCH

            out = next_layer([input_tensor, out])

        else:
            next_layer = Conv2D(
                self._filters[-1],
                1, 
                strides=(self._stride1, self._stride1),
                kernel_initializer=self._initializer,
                name=f'{self._block_name}_conv_{ResnetBlock.skip_identifier}')
            next_layer._gloro_branch = gloro.constants.SKIP_BRANCH

            conv_skip = next_layer(input_tensor)

            next_layer = Add(
                name=f'{self._block_name}_{ResnetBlock.join_identifier}')
            next_layer._gloro_branch = gloro.constants.MAIN_BRANCH

            out = next_layer([conv_skip, out])

        next_layer = self._activation(None)
        next_layer._gloro_branch = gloro.constants.MAIN_BRANCH

        return next_layer(out)
