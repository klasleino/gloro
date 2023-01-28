import numpy as np

from pytest import mark

from gloro.lc import PowerMethod


class ModelLipschitzConstants:

    def test_small_orthonormal_dense_untrained_sub(self, models):
        sub_lc = self.tensors.to_np(
            models.orthonormal_dense(lc_strategy='eigh').sub_lipschitz
        )
        assert np.allclose(sub_lc, 1.)

    # These tests make sure that the norm of the Jacobian is upper-bounded by
    # the Lipschitz constant at various random sample points.

    def test_jacobian_small_orthonormal_dense(self, models):
        gloronet = models.orthonormal_dense(lc_strategy='eigh')

        X = self.tensors.random((8, *models.orthonormal_dense.input_shape))

        J = self.tensors.to_np(self.tensors.margin_jacobian(gloronet.f, X))

        lower_bound = np.linalg.norm(J, ord=2, axis=-1)

        lc = self.tensors.to_np(gloronet.lipschitz_constant(X))
        lc = np.where(lc == -1, 0, lc)

        assert np.all(lower_bound <= lc + 1e-6)

    def test_jacobian_random_conv(self, models):
        gloronet = models.random_conv(lc_strategy=PowerMethod(25))

        X = self.tensors.random((8, *models.random_conv.input_shape))

        J = self.tensors.to_np(self.tensors.margin_jacobian(gloronet.f, X))

        lower_bound = np.linalg.norm(J, ord=2, axis=-1)

        lc = self.tensors.to_np(gloronet.lipschitz_constant(X))
        lc = np.where(lc == -1, 0, lc)

        assert np.all(lower_bound <= lc + 1e-6)
