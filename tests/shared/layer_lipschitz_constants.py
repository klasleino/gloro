import numpy as np

from pytest import mark

from gloro.lc import PowerMethod


class LayerLipschitzConstants:

    def test_dense_layer_orthogonal(self, layers):
        lc = self.tensors.to_np(
            layers.orthonormal_dense(lc_strategy=PowerMethod(100)).lipschitz()
        )
        assert np.allclose(lc, 1.)

    def test_dense_layer_orthogonal_eigh(self, layers):
        lc = self.tensors.to_np(
            layers.orthonormal_dense(lc_strategy='eigh').lipschitz()
        )
        assert np.allclose(lc, 1.)

    def test_average_pooling_layer(self, layers):
        assert np.allclose(
            self.tensors.to_np(layers.average_pooling(pool_size=2).lipschitz()),
            0.5,
        )

    def test_scaling_layer(self, layers):
        assert np.allclose(
            self.tensors.to_np(layers.scaling(scale=4.).lipschitz()), 4.
        )

    # For sanity checks, just make sure that the Lipschitz constant is an upper
    # bound on the distortion of a sample. For these tests, repeat a few times,
    # since there is randomness involved.

    def _sample_lc(self, layer, input_shape):
        X1 = self.tensors.random((1, *input_shape))
        Z1 = layer(X1)

        X2 = self.tensors.random((1, *input_shape))
        Z2 = layer(X2)

        return self.tensors.to_np(
            self.tensors.norm(Z1 - Z2) / self.tensors.norm(X1 - X2)
        )

    @mark.parametrize('i', range(5))
    def test_dense_layer_computer_sanity(self, layers, i):
        layer = layers.random_dense(lc_strategy=PowerMethod(100))

        lc = self.tensors.to_np(layer.lipschitz())

        assert self._sample_lc(layer, layers.random_dense.input_shape) <= lc

    @mark.parametrize('i', range(5))
    def test_conv_layer_computer_sanity(self, layers, i):
        layer = layers.random_conv(lc_strategy=PowerMethod(100))

        lc = self.tensors.to_np(layer.lipschitz())

        assert self._sample_lc(layer, layers.random_conv.input_shape) <= lc

    @mark.parametrize('i', range(5))
    def test_conv_layer_computer_sanity_strides(self, layers, i):
        layer = layers.strided_conv(lc_strategy=PowerMethod(100))

        lc = self.tensors.to_np(layer.lipschitz())

        assert self._sample_lc(layer, layers.strided_conv.input_shape) <= lc
