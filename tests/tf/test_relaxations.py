import numpy as np
import tensorflow as tf

from pytest import fixture
from sklearn.datasets import make_moons

from .injection.tensors import Tensors

from gloro import GloroNet
from gloro import RtkGloroNet
from gloro import AffinityGloroNet
from gloro.layers import Dense
from gloro.layers import Input
from gloro.layers import MinMax
from gloro.training.callbacks import LrScheduler


class TestRelaxations:

    @fixture(autouse=True)
    def tensors(self):
        self.tensors = Tensors()
        yield self.tensors


    def _get_synthetic_data(self):
        def get_blue(n=100, std=0.25):
            x = np.arange(2.5, 5.75, 3.25 / n)
            y = -0.5* (x - 4.75)**2 + 2.0

            return (
                np.concatenate((x[:,None],y[:,None]), axis=1) +
                np.random.normal(0., std / 2, [n, 2])
            )

        def get_green(n=100, std=0.25):
            x = np.arange(5.0, 7.5, 2.5 / (n // 2))
            y = 0.5*(x - 5.5)**2 + 1.0

            return np.concatenate(
                (
                    np.concatenate((x[:,None],y[:,None]), axis=1) +
                        np.random.normal(0., std / 2, [n//2, 2]),
                    np.random.normal([7.0, 0.0], [std, std], [n//2, 2])
                ),
                axis=0,
            )

        def get_red(n=100, std=0.25):
            x = np.arange(-0.5, 3.5, 4.0 / n)
            y = 0.5 * (x - 1.0)**2 - 2.

            return (
                np.concatenate((x[:,None],y[:,None]), axis=1) +
                np.random.normal(0., std / 2, [n, 2])
            )

        def get_yellow(n=100, std=0.25):
            x = np.arange(-2.5, 1.0, 3.5 / (n // 2))
            y = -0.5*(x+0.5)**2 - 0.0

            return np.concatenate(
                (
                    np.concatenate((x[:,None],y[:,None]), axis=1) +
                        np.random.normal(0., std / 2, [n//2, 2]),
                    np.random.normal([-0.75, 1.0], [std, std], [n//2, 2])
                ),
                axis=0,
            )

        X = np.concatenate(
            (
                get_blue(),
                get_green(),
                get_red(),
                get_yellow(),
            ),
            axis=0,
        )

        Y = np.concatenate((
            0 * np.ones((100), 'int32'),
            1 * np.ones((100), 'int32'),
            2 * np.ones((100), 'int32'),
            3 * np.ones((100), 'int32'))
        )

        shuffle = np.random.permutation(400)

        X = X[shuffle]
        Y = Y[shuffle]

        return X, Y


    def test_train_rtk(self, tmp_path):
        # Create a synthetic dataset.
        X, Y = self._get_synthetic_data()

        # Make the model.
        x = Input((2,))
        z = Dense(100)(x)
        z = MinMax()(z)
        z = Dense(100)(z)
        z = MinMax()(z)
        y = Dense(4)(z)

        g_rt2 = RtkGloroNet(x, y, 0.2, k=2)

        # Compile and train.
        g_rt2.compile(
            loss='sparse_crossentropy',
            optimizer='adam',
            metrics=['clean_acc', 'vra', 'rtk_vra'],
        )
        g_rt2.fit(
            X, Y,
            epochs=5,
            batch_size=8,
            callbacks=[LrScheduler('[1.]-cos-[0.]')],
        )

        loss, clean_acc, vra, rtk_vra = g_rt2.evaluate(X, Y, batch_size=32)

        # Check some basic stats that indicate the success and correctness of
        # training.

        assert vra <= rtk_vra
        assert vra <= clean_acc

        # Proves we actually learned something during training.
        assert vra > 0.80
        assert rtk_vra > 0.85

        # Saving and reloading should work, and the loaded model should achieve
        # the same stats.

        file_name = f'{tmp_path}/model_rtk.gloronet'

        g_rt2.save(file_name)

        g2 = RtkGloroNet.load_model(file_name)

        # We still have to recompile the new model since the loss isn't saved.
        g2.compile(
            loss='sparse_crossentropy',
            optimizer='adam',
            metrics=['clean_acc', 'vra', 'rtk_vra'],
        )

        loss2, clean_acc2, vra2, rtk_vra2 = g2.evaluate(X, Y, batch_size=32)

        assert np.allclose(
            [vra, clean_acc, rtk_vra],
            [vra2, clean_acc2, rtk_vra2]
        )


    def test_train_affinity(self, tmp_path):
        # Create a synthetic dataset.
        X, Y = self._get_synthetic_data()

        # Make the model.
        x = Input((2,))
        z = Dense(100)(x)
        z = MinMax()(z)
        z = Dense(100)(z)
        z = MinMax()(z)
        y = Dense(4)(z)

        g_rb = AffinityGloroNet(x, y, 0.2, affinity_sets=[[0,2],[1],[3]])

        # Compile and train.
        g_rb.compile(
            loss='sparse_crossentropy',
            optimizer='adam',
            metrics=['clean_acc', 'vra', 'affinity_vra'],
        )
        g_rb.fit(
            X, Y,
            epochs=10,
            batch_size=8,
        )

        _, clean_acc, vra, affinity_vra = g_rb.evaluate(X, Y, batch_size=32)

        # Check some basic stats that indicate the success and correctness of
        # training.

        assert vra <= affinity_vra
        assert vra <= clean_acc

        # Proves we actually learned something during training.
        assert vra > 0.75
        assert affinity_vra > 0.78

        # Saving and reloading should work, and the loaded model should achieve
        # the same stats.

        file_name = f'{tmp_path}/model_affinity.gloronet'

        g_rb.save(file_name)

        g2 = AffinityGloroNet.load_model(file_name)

        # We still have to recompile the new model since the loss isn't saved.
        g2.compile(
            loss='sparse_crossentropy',
            optimizer='adam',
            metrics=['clean_acc', 'vra', 'affinity_vra'],
        )

        _, clean_acc2, vra2, affinity_vra2 = g2.evaluate(X, Y, batch_size=32)

        assert np.allclose(
            [vra, clean_acc, affinity_vra],
            [vra2, clean_acc2, affinity_vra2]
        )
