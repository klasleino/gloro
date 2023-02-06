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

    @mark.parametrize('kernel_size', [1,3,4,5,6])
    def test_liresnet_block_identity(self, layers, kernel_size):
        layer = layers.liresnet_block_identity(
            kernel_size=kernel_size,
            lc_strategy=PowerMethod(100),
        )

        # Make sure running the layer forward actually gives you the same input.
        X = self.tensors.random(
            (1, *layers.liresnet_block_identity.input_shape)
        )
        Z = layer(X)

        assert np.allclose(
            self.tensors.to_np(X), self.tensors.to_np(Z),
            atol=1e-06
        )

        # Make sure the Lipschitz constant is 1, since this is the identity.
        lc = self.tensors.to_np(layer.lipschitz())

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

    @mark.parametrize('i', range(5))
    def test_liresnet_block_no_affine(self, layers, i):
        layer = layers.liresnet_block_no_affine(lc_strategy=PowerMethod(100))

        lc = self.tensors.to_np(layer.lipschitz())
        sample_lc = self._sample_lc(
            layer, layers.liresnet_block_no_affine.input_shape
        )
        assert sample_lc <= lc

    @mark.parametrize('i', range(5))
    def test_liresnet_block_with_affine(self, layers, i):
        layer = layers.liresnet_block_with_affine(lc_strategy=PowerMethod(100))

        lc = self.tensors.to_np(layer.lipschitz())
        sample_lc = self._sample_lc(
            layer, layers.liresnet_block_with_affine.input_shape
        )
        assert sample_lc <= lc
