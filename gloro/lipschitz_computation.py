import tensorflow as tf

from tensorflow.keras.layers import Add
from tensorflow.keras.layers import AveragePooling2D

import gloro

from gloro.layers.network_layers import ResnetBlock
from gloro.utils import l2_normalize


class LipschitzComputer(object):

    def __init__(self, layer, *args, **kwargs):
        self._name = layer.name

        if hasattr(layer, '_gloro_branch'):
            self._branch = layer._gloro_branch

        elif layer.name.startswith(ResnetBlock.identifier):
            # TODO: this is a little less nice than reading a `_gloro_branch` 
            #   property, but it persists by default when the layers are saved,
            #   whereas we would need extra instrumentation to save the
            #   `_gloro_branch` property. Ultimately we should probably pick
            #   just one method (either name-based or property-based).
            if ResnetBlock.join_identifier in layer.name:
                self._branch = gloro.constants.MAIN_BRANCH

            elif ResnetBlock.skip_identifier in layer.name:
                self._branch = gloro.constants.SKIP_BRANCH

            else:
                self._branch = gloro.constants.RESIDUAL_BRANCH

        else:
            self._branch = gloro.constants.MAIN_BRANCH

    @property
    def name(self):
        return self._name

    @property
    def branch(self):
        return self._branch

    @staticmethod
    def for_layer(layer, num_iterations):
        if hasattr(layer, 'kernel'):
            if len(layer.kernel.shape) == 4:
                return ConvLayerComputer(layer, num_iterations)

            else:
                return DenseLayerComputer(layer, num_iterations)

        elif isinstance(layer, gloro.layers.Scaling):
            return ScalingLayerComputer(layer)

        elif isinstance(layer, Add):
            return JoinLayerComputer(layer)

        elif isinstance(layer, AveragePooling2D):
            return AveragePoolingComputer(layer)

        else:
            return LipschitzComputer(layer)

    @staticmethod
    def for_model(model, num_iterations, exclude_last_layer=True):
        layers = model.layers[:-1] if exclude_last_layer else model.layers

        return [
            LipschitzComputer.for_layer(layer, num_iterations)
            for layer in layers
        ]

    @staticmethod
    def global_lipschitz_bound(layer_computers):
        lc = {
            gloro.constants.MAIN_BRANCH: 1.,
            gloro.constants.RESIDUAL_BRANCH: 1.,
            gloro.constants.SKIP_BRANCH: 1.,
        }

        for layer in layer_computers:
            lc[layer.branch] *= layer.get_lipschitz_constant(lc=lc)

        return lc[gloro.constants.MAIN_BRANCH]

    def get_lipschitz_constant(self, **kwargs):
        return 1.


class DenseLayerComputer(LipschitzComputer):
    def __init__(self, layer, num_iterations):
        super().__init__(layer)

        self._W = layer.kernel
        self._iterate = tf.Variable(
            tf.random.truncated_normal((layer.kernel.shape[1], 1)),
            dtype='float32',
            trainable=False)

        self._while_cond = lambda i, _: i < num_iterations

    @property
    def W(self):
        return self._W

    @property
    def iterate(self):
        return self._iterate

    def get_lipschitz_constant(self, **kwargs):

        def body(i, x):
            x = l2_normalize(x)
            x_p = self.W @ x
            x = tf.transpose(self.W) @ x_p

            return i + 1, x

        _, x = tf.while_loop(
            self._while_cond, body, [tf.constant(0), self.iterate])

        # Update the power iterate.
        self.iterate.assign(x)

        return tf.sqrt(
            tf.reduce_sum((self.W @ x)**2.) / 
            (tf.reduce_sum(x**2.) + gloro.constants.EPS))


class ConvLayerComputer(LipschitzComputer):
    def __init__(self, layer, num_iterations):
        super().__init__(layer)

        self._W = layer.kernel
        self._strides = layer.strides
        self._padding = layer.padding.upper()
        self._iterate = tf.Variable(
            tf.random.truncated_normal((1, *layer.input_shape[1:])),
            dtype='float32',
            trainable=False)

        self._while_cond = lambda i, _: i < num_iterations

    @property
    def W(self):
        return self._W

    @property
    def iterate(self):
        return self._iterate

    @property
    def strides(self):
        return self._strides

    @property
    def padding(self):
        return self._padding

    def get_lipschitz_constant(self, **kwargs):

        def body(i, x):
            x = l2_normalize(x)
            x_p = tf.nn.conv2d(
                x,
                self.W,
                strides=self.strides,
                padding=self.padding)
            x = tf.nn.conv2d_transpose(
                x_p,
                self.W,
                x.shape,
                strides=self.strides,
                padding=self.padding)

            return i + 1, x

        _, x = tf.while_loop(
            self._while_cond, body, [tf.constant(0), self._iterate])

        # Update the power iterate.
        self.iterate.assign(x)

        Wx = tf.nn.conv2d(x, self.W, strides=self.strides, padding=self.padding)

        return tf.sqrt(
            tf.reduce_sum(Wx**2.) / 
            (tf.reduce_sum(x**2.) + gloro.constants.EPS))


class ScalingLayerComputer(LipschitzComputer):
    def __init__(self, layer):
        super().__init__(layer)

        self._w = layer._weight

    @property
    def w(self):
        return self._w

    def get_lipschitz_constant(self, **kwargs):
        return tf.abs(self.w)


class JoinLayerComputer(LipschitzComputer):

    def get_lipschitz_constant(self, lc):
        result = (
            lc[gloro.constants.RESIDUAL_BRANCH] + 
            lc[gloro.constants.SKIP_BRANCH])

        lc[gloro.constants.RESIDUAL_BRANCH] = 1.
        lc[gloro.constants.SKIP_BRANCH] = 1.

        return result


class AveragePoolingComputer(LipschitzComputer):
    def __init__(self, layer):
        super().__init__(layer)

        W = tf.eye(layer.input.shape[-1])[None,None] * (
            tf.ones(layer.pool_size)[:,:,None,None]) / (
                layer.pool_size[0] * layer.pool_size[1])

        x0 = tf.random.truncated_normal(
            shape=(1,*layer.input_shape[1:]))

        def body(i, x):
            x = l2_normalize(x)
            x_p = tf.nn.conv2d(
                x, W,
                strides=layer.strides,
                padding=layer.padding.upper())
            x = tf.nn.conv2d_transpose(
                x_p, W, x.shape,
                strides=layer.strides,
                padding=layer.padding.upper())

            return i + 1, x

        _, x = tf.while_loop(lambda i, _: i < 100, body, [tf.constant(0), x0])

        Wx = tf.nn.conv2d(
            x, W, 
            strides=layer.strides, 
            padding=layer.padding.upper())

        self._lc = tf.sqrt(
            tf.reduce_sum(Wx**2.) / 
            (tf.reduce_sum(x**2.) + gloro.constants.EPS))

    def get_lipschitz_constant(self, **kwargs):
        return self._lc
