import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from keras.layers import Lambda

import gloro

from gloro.models import GloroNet
from gloro.relaxations.affinity_sets import AffinitySet
from gloro.training.losses import get as get_loss
from gloro.utils import get_value
from gloro.utils import set_value


class RtkGloroNet(GloroNet):
    """
    TODO: docstring.
    """
    def __init__(
        self, 
        inputs=None, 
        outputs=None, 
        epsilon=None, 
        k=None,
        *,
        model=None, 
        **kwargs,
    ):
        super().__init__(inputs, outputs, epsilon, model=model, **kwargs)

        if k is None:
            raise ValueError('`k` is required')

        self._k = tf.Variable(k, dtype='int32', name='k', trainable=False)

        self.output_names = ['pred', 'guarantee', 'pred_top_k']


    @property
    def k(self):
        return get_value(self._k)

    @k.setter
    def k(self, new_value):
        set_value(self._k, new_value)


    # Prediction variations.

    # TODO(klas): In theory, we could provide the maximum radius for each k in
    #   [K] for which X is top-k robust. Or, we could provide the maximum of any
    #   such radius. At any rate, the implementation of this method in the
    #   `GloroNet` superclass gives a radius that is only meaningful for top-1
    #   robustness, so we are removing this method for now. In the future we may
    #   implement an RTK variation.
    def certified_radius(self, X):
        raise NotImplementedError(
            '`certified_radius` is not currently implemented for `RtkGloroNet`'
        )

    def predict_with_certified_radius(self, X):
        raise NotImplementedError(
            '`predict_with_certified_radius` is not currently implemented for '
            '`RtkGloroNet`'
        )

    def predict_with_guarantee(self, *args, **kwargs):
        return super().predict(*args, **kwargs)[0:2]

    def predict_with_loosest_guarantee(self, *args, **kwargs):
        preds, guarantees = super().predict(*args, **kwargs)[0:2]

        guarantees = 1 + tf.argmax(
            tf.cast(guarantees, 'float32') +
                # Adding this breaks ties in favor of larger k.
                tf.cast(tf.range(self.k), 'float32') * 1e-7,
            axis=1)

        # Guarantees are only valid on robust points. Otherwise indicate that
        # no guarantee was achieved with -1.
        guarantees = tf.where(
            tf.equal(tf.argmax(preds, axis=1), preds.shape[1] - 1),
            -1,
            guarantees)

        return [preds, guarantees.numpy()]

    def predict_with_strictest_guarantee(self, *args, **kwargs):
        preds, guarantees = super().predict(*args, **kwargs)[0:2]

        guarantees = 1 + tf.argmax(guarantees, axis=1).numpy()

        # Guarantees are only valid on robust points. Otherwise indicate that
        # no guarantee was achieved with -1.
        guarantees = tf.where(
            tf.equal(tf.argmax(preds, axis=1), preds.shape[1] - 1),
            -1,
            guarantees)

        return [preds, guarantees.numpy()]


    # Margin and Lipschitz computation.

    def _K_ij(self, sub_lipschitz, j):
        """
        K_ij is the Lipschitz constant on the margin y_j - y_i.
        """
        kW = sub_lipschitz * self.layers[-1].kernel

        # Get the weight column of the predicted class.
        kW_j = tf.gather(tf.transpose(kW), j)

        # Get weights that predict the value y_j - y_i for all i != j.
        kW_ij = kW_j[:,:,:,None] - kW[None,None]

        return tf.sqrt(tf.reduce_sum(kW_ij * kW_ij, axis=2))

    def _mk(self, sub_lipschitz, y):
        """
        For each k in [K] mk is the maximum margin by which the kth-highest
        logit surpasses all logits not in the top k.
        """

        # For k in [K], j is the kth highest class, and y_j is the corresponding
        # logit value.
        y_j, j = tf.math.top_k(y, k=self._k)

        # K_ij is the Lipschitz constant on the margin y_j - y_i.
        K_ij = self._K_ij(sub_lipschitz, j)

        where_in_top_k = tf.equal(y[:,None], y_j[:,:,None])
        where_yi_lt_yjk = tf.math.cumsum(
            tf.where(where_in_top_k, -np.inf, 0.), axis=1
        )
        where_l_leq_k = tf.where(
            tf.equal(tf.cumsum(tf.eye(tf.cast(self._k, 'uint8'))), 0.),
            -np.inf,
            0,
        )

        mk_j = y_j[:,None] - tf.reduce_max(
            y[:,None,None] + self._epsilon * K_ij[:,None] +
                where_yi_lt_yjk[:,:,None] +
                where_l_leq_k[None,:,:,None],
            axis=3,
        )

        return tf.reduce_min(mk_j, axis=2)

    def lipschitz_constant(self, X=None, y=None):
        if y is None:
            if X is None:
                y = tf.eye(self.num_classes)
            else:
                y = self.f(X)

        # For k in [K], j is the kth highest class, and y_j is the corresponding
        # logit value.
        y_j, j = tf.math.top_k(y, k=self._k)

        # K_ij is the Lipschitz constant on the margin y_j - y_i.
        K_ij = self._K_ij(self.sub_lipschitz, j)

        return K_ij + tf.math.cumsum(
            tf.where(
                tf.equal(y[:,None], y_j[:,:,None]),
                -np.inf,
                0.),
            axis=1,
            reverse=True,
        )


    # Overriding `keras.Model` functions.

    def call(self, X, training=False):
        y = self.f(X, training=training)

        sub_lipschitz = self.sub_lipschitz

        # For each k in [K] mk is the maximum margin by which the kth-highest
        # logit surpasses all logits not in the top k.
        mk = self._mk(sub_lipschitz, y)

        # m is the the largest margin by which some k in [K] surpasses all other
        # logits not in the top k.
        m = tf.reduce_max(mk, axis=1)

        # We also keep track of all k in [K] for which the model is top-k
        # robust.
        k_guarantee = tf.cast(mk > 0, 'int32')

        # For the purpose of metrics, get the most lax (i.e., largest) k that we
        # are top-k robust with. Specifically, this is used for top-k VRA.
        loosest_k = tf.cast(
            1 + tf.argmax(
                tf.cast(mk > 0, 'float32') +
                    # Adding this breaks ties in favor of larger k.
                    tf.cast(tf.range(self._k), 'float32') * 1e-7,
                axis=1),
            'float32'
        )[:,None]

        y_bot = tf.reduce_max(y, axis=1) - m

        y_with_bot = tf.concat([y, y_bot[:,None]], axis=1)

        # We name the different outputs for use with the loss function and
        # metrics.
        return [
            tf.identity(y_with_bot, name='pred'),
            tf.identity(k_guarantee, name='guarantee'),
            tf.concat([y_with_bot, loosest_k], axis=1, name='pred_top_k'),
        ]

    def predict(self, *args, **kwargs):
        return super().predict(*args, **kwargs)[0]

    def evaluate(self, *args, **kwargs):
        return super().evaluate(*args, **kwargs)[1:]

    def compile(self, **kwargs):
        if 'loss' in kwargs:
            if isinstance(kwargs['loss'], str):
                kwargs['loss'] = {'pred': get_loss(kwargs['loss'])}
            else:
                kwargs['loss'] = {'pred': kwargs['loss']}

        if 'metrics' in kwargs:
            metrics = kwargs['metrics']
            new_metrics = []

            includes_rtk_vra = False
            rtk_vra = None
            for metric in metrics:
                if metric in gloro.constants.GLORO_CUSTOM_OBJECTS:
                    metric = gloro.constants.GLORO_CUSTOM_OBJECTS[metric]

                if (
                    hasattr(metric, '__name__') and (
                        metric.__name__.startswith('rtk_vra') or
                        metric.__name__.startswith('affinity_vra'))
                ):
                    includes_rtk_vra = True
                    rtk_vra = metric

                else:
                    new_metrics.append(metric)

            kwargs['metrics'] = {'pred': new_metrics}

            if includes_rtk_vra:
                kwargs['metrics']['pred_top_k'] = rtk_vra

        Model.compile(self, **kwargs)

    def get_config(self):
        config = {
            'k': int(self.k),
        }
        base_config = super().get_config()

        return dict(list(base_config.items()) + list(config.items()))


class AffinityGloroNet(RtkGloroNet):
    """
    TODO: docstring.
    """
    def __init__(
        self,
        inputs=None,
        outputs=None,
        epsilon=None,
        affinity_sets=None,
        *,
        model=None,
        **kwargs,
    ):
        if affinity_sets is None:
            raise ValueError('`affinity_sets` is required')

        # Parse the provided affinity sets.
        if isinstance(affinity_sets, AffinitySet):
            pass

        elif isinstance(affinity_sets, (list, tuple)):
            affinity_sets = AffinitySet.from_class_indices(affinity_sets)

        elif isinstance(affinity_sets, np.ndarray):
            affinity_sets = AffinitySet.from_one_hot(affinity_sets)

        elif isinstance(affinity_sets, str):
            affinity_sets = AffinitySet.from_string(affinity_sets)

        else:
            raise ValueError(
                f'unexpected type for `affinity_sets`: {type(affinity_sets)}')

        k = affinity_sets.max_set_size

        super().__init__(inputs, outputs, epsilon, k, model=model, **kwargs)

        # Make sure that the given affinity sets match the model's output shape.
        if affinity_sets.mask.shape[1] != self.num_classes:
            raise ValueError(
                f'affinity sets defined for a number of classes that does not '
                f'match the output shape of the model: '
                f'{affinity_sets.mask.shape[1]} vs {self.num_classes}'
            )

        self._affinity_sets = affinity_sets


    @property
    def affinity_sets(self):
        return self._affinity_sets

    @property
    def k(self):
        return get_value(self._k)

    @k.setter
    def k(self, new_value):
        raise ValueError('`k` cannot be set on `AffinityGloroNet`')


    # Prediction variations.

    # TODO(klas): In theory, we could provide the maximum radius for which X is
    #    affinity-robust. At any rate, the implementation of this method in the
    #   `GloroNet` superclass gives a radius that is only meaningful for top-1
    #   robustness, so we are removing this method for now. In the future we may
    #   implement an affinity variation.
    def certified_radius(self, X):
        raise NotImplementedError(
            '`certified_radius` is not currently implemented for '
            '`AffinityGloroNet'
        )

    def predict_with_certified_radius(self, X):
        raise NotImplementedError(
            '`predict_with_certified_radius` is not currently implemented for '
            '`AffinityGloroNet`'
        )


    # Overriding `RtkGloroNet` helpers.

    def _mk(self, sub_lipschitz, y):
        """
        For each k in [K] mk is the maximum margin by which the kth-highest
        logit surpasses all logits not in the top k, if the top k classes all
        belong to the same affinity set (and -infinity otherwise).
        """

        # For k in [K], j is the kth highest class, and y_j is the corresponding
        # logit value.
        y_j, j = tf.math.top_k(y, k=self._k)

        # K_ij is the Lipschitz constant on the margin y_j - y_i.
        K_ij = self._K_ij(sub_lipschitz, j)

        where_in_top_k = tf.equal(y[:,None], y_j[:,:,None])
        where_yi_lt_yjk = tf.math.cumsum(
            tf.where(where_in_top_k, -np.inf, 0.), axis=1
        )
        where_l_leq_k = tf.where(
            tf.equal(tf.cumsum(tf.eye(tf.cast(self._k, 'uint8'))), 0.),
            -np.inf,
            0,
        )

        mk_j = y_j[:,None] - tf.reduce_max(
            y[:,None,None] + self._epsilon * K_ij[:,None] +
                where_yi_lt_yjk[:,:,None] +
                where_l_leq_k[None,:,:,None],
            axis=3,
        )

        mk = tf.reduce_min(mk_j, axis=2)

        # We don't want to consider any margins that correspond to a k that
        # corresponds to a set of classes that do not match any of our affinity
        # sets.
        return tf.where(
            self._affinity_sets.where_matches(where_in_top_k),
            mk,
            -np.inf,
        )


    def get_config(self):
        config = {
            'affinity_sets': str(self.affinity_sets),
        }
        base_config = super().get_config()

        # Note that `k` is not passed directly to the constructor like it is in
        # `RtkGloroNet`.
        del base_config['k']

        return dict(list(base_config.items()) + list(config.items()))
