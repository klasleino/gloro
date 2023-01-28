import numpy as np
import tensorflow as tf

from pytest import fixture
from sklearn.datasets import make_moons

from .injection.tensors import Tensors

from gloro import GloroNet
from gloro.lc import PowerMethod
from gloro.layers import Dense
from gloro.layers import Input
from gloro.layers import MinMax


class TestBasicTraining:

    @fixture(autouse=True)
    def tensors(self):
        self.tensors = Tensors()
        yield self.tensors

    def test_train_two_moons(self, tmp_path):
        # Make data.
        X, Y = make_moons(n_samples=200, noise=0.05)

        # Make the model.
        x = Input((2,))
        z = Dense(100, lc_strategy='eigh')(x)
        z = MinMax()(z)
        z = Dense(100)(z)
        z = MinMax()(z)
        y = Dense(2)(z)

        g = GloroNet(x, y, 0.1)

        # Compile and train.
        g.compile(
            loss='sparse_crossentropy',
            optimizer='adam',
            metrics=['vra', 'clean_acc'],
        )
        g.fit(X, Y, epochs=5, batch_size=8)

        loss, vra, clean_acc = g.evaluate(X, Y, batch_size=8)

        # Check some basic stats that indicate the success and correctness of
        # training.

        assert vra <= clean_acc

        # We should be able to get at least 75% VRA, which proves we actually
        # learned something during training.
        assert vra > 0.75

        # Check that the Lipschitz constant is a valid upper bound.
        X_sample = tf.constant(X[np.random.randint(0, 200, 20)])

        J = self.tensors.to_np(self.tensors.margin_jacobian(g.f, X_sample))

        lower_bound = np.linalg.norm(J, ord=2, axis=-1)

        lc = self.tensors.to_np(g.lipschitz_constant(X_sample))
        lc = np.where(lc == -1, 0, lc)

        assert np.all(lower_bound <= lc + 1e-6)

        # Saving and reloading should work, and the loaded model should achieve
        # the same stats.

        file_name = f'{tmp_path}/model.gloronet'

        g.save(file_name)

        g2 = GloroNet.load_model(file_name)

        # We still have to recompile the new model since the loss isn't saved.
        g2.compile(
            loss='sparse_crossentropy',
            optimizer='adam',
            metrics=['vra', 'clean_acc'],
        )

        loss2, vra2, clean_acc2 = g2.evaluate(X, Y, batch_size=8)

        assert np.allclose([vra, clean_acc], [vra2, clean_acc2])
