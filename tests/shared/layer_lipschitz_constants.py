import numpy as np

from pytest import mark

from gloro.lipschitz_computation import ConvLayerComputer
from gloro.lipschitz_computation import DenseLayerComputer

class LayerLipschitzConstants:

    def test_dense_layer_computer_orthogonal(self, layers):
        lc = self.tensors.to_np(
            DenseLayerComputer(layers.orthonormal_dense(), 100)
                .get_lipschitz_constant()
        )
        assert np.allclose(lc, 1.)

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
        layer = layers.random_dense()

        lc = self.tensors.to_np(
            DenseLayerComputer(layer, 100).get_lipschitz_constant()
        )
        assert self._sample_lc(layer, layers.random_dense.input_shape) <= lc

    @mark.parametrize('i', range(5))
    def test_conv_layer_computer_sanity(self, layers, i):
        layer = layers.random_conv()

        lc = self.tensors.to_np(
            ConvLayerComputer(layer, 25).get_lipschitz_constant()
        )
        assert self._sample_lc(layer, layers.random_conv.input_shape) <= lc

    @mark.parametrize('i', range(5))
    def test_conv_layer_computer_sanity_strides(self, layers, i):
        layer = layers.strided_conv()

        lc = self.tensors.to_np(
            ConvLayerComputer(layer, 25).get_lipschitz_constant()
        )
        assert self._sample_lc(layer, layers.strided_conv.input_shape) <= lc
