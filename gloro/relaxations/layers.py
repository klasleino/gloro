import numpy as np
import tensorflow as tf

from gloro.layers.margin_layers import LipschitzMargin
from gloro.utils import get_value
from gloro.utils import set_value


class LipschitzMarginRtk(LipschitzMargin):
    def __init__(
        self, 
        epsilon,
        k,
        model=None,
        num_iterations=5, 
        norm='l2', 
        strictest_guarantee=False,
        **kwargs,
    ):
        super().__init__(
            epsilon, 
            model=model,
            num_iterations=num_iterations, 
            norm=norm, 
            **kwargs)

        self._k = tf.Variable(k, name='k', trainable=False)
        self._strictest_guarantee = tf.Variable(
            strictest_guarantee, name='strict', trainable=False)

    @property
    def k(self):
        return get_value(self._k)

    @k.setter
    def k(self, new_k):
        set_value(self._k, new_k)

    @property
    def strictest_guarantee(self):
        return get_value(self._strictest_guarantee)

    @strictest_guarantee.setter
    def strictest_guarantee(self, new_strictest_guarantee):
        set_value(self._strictest_guarantee, new_strictest_guarantee)

    def _get_yj_j_Kij(self, y):

        kW = self._kW()

        y_j, j = tf.math.top_k(y, k=self._k)
        where_not_j = tf.not_equal(y[:,None], y_j[:,:,None])

        # Get the weight column of the predicted class.
        kW_j = tf.gather(tf.transpose(kW), j)

        # Get weights that predict the value y_j - y_i for all i != j.
        kW_ij = kW_j[:,:,:,None] - kW[None,None]

        # We do this instead of `tf.linalg.norm(W_d)` because of an apparent bug
        # in `tf.linalg.norm` that leads to NaN values.
        K_ij = tf.sqrt(tf.reduce_sum(kW_ij * kW_ij, axis=2))

        return y_j, j, K_ij

    def _lipschitz_constant_fn(self, y):

        y_j, j, K_ij = self._get_yj_j_Kij(y)

        return K_ij + tf.math.cumsum(
            tf.where(
                tf.equal(y[:,None], y_j[:,:,None]), 
                -np.inf, 
                0.), 
            axis=1, 
            reverse=True)
      
    def _get_mk(self, y):
        y_j, j, K_ij = self._get_yj_j_Kij(y)
        
        where_in_top_k = tf.equal(y[:,None], y_j[:,:,None])
        where_yi_lt_yjk = tf.math.cumsum(
            tf.where(where_in_top_k, -np.inf, 0.), 
            axis=1)
        where_l_leq_k = tf.where(
            tf.equal(tf.cumsum(tf.eye(tf.cast(self._k, 'uint8'))), 0.),
            -np.inf,
            0)
        
        mk_j = y_j[:,None] - tf.reduce_max(
          y[:,None,None] + self._epsilon * K_ij[:,None] + 
              where_yi_lt_yjk[:,:,None] + 
              where_l_leq_k[None,:,:,None],
          axis=3)
        
        return tf.reduce_min(mk_j, axis=2)
      
    def call(self, y):
      
        mk = self._get_mk(y)
        
        m = tf.reduce_max(mk, axis=1)

        all_guarantees = tf.cast(mk > 0, 'int32')

        # For the purpose of metrics, get the most lax (i.e., largest) k that we
        # are top-k robust with.
        loosest_k = tf.cast(
            1 + tf.argmax(
                tf.cast(mk > 0, 'float32') +
                    # Adding this breaks ties in favor of larger k.
                    tf.cast(tf.range(self._k), 'float32') * 1e-7,
                axis=1),
            'float32')[:,None]

        y_bot = tf.stop_gradient(tf.reduce_max(y, axis=1)) - m

        return tf.concat([y, y_bot[:,None]], axis=1), all_guarantees, loosest_k


class LipschitzMarginAffinity(LipschitzMarginRtk):
    def __init__(
        self, 
        epsilon,
        k,
        affinity_sets,
        model=None,
        num_iterations=5, 
        norm='l2', 
        **kwargs,
    ):
        super().__init__(
            epsilon, 
            k=k,
            model=model,
            num_iterations=num_iterations, 
            norm=norm, 
            **kwargs)

        self._affinity_sets = affinity_sets

    @property
    def k(self):
        return super().k

    @k.setter
    def k(self, new_k):
        raise RuntimeError('`k` is not settable')

    @property
    def affinity_sets(self):
        return self._affinity_sets

    def _get_mk(self, y):
        y_j, j, K_ij = self._get_yj_j_Kij(y)
        
        where_in_top_k = tf.equal(y[:,None], y_j[:,:,None])
        where_yi_lt_yjk = tf.math.cumsum(
            tf.where(where_in_top_k, -np.inf, 0.), 
            axis=1)
        where_l_leq_k = tf.where(
            tf.equal(tf.cumsum(tf.eye(tf.cast(self._k, 'uint8'))), 0.),
            -np.inf,
            0)
        
        mk_j = y_j[:,None] - tf.reduce_max(
          y[:,None,None] + self._epsilon * K_ij[:,None] + 
              where_yi_lt_yjk[:,:,None] + 
              where_l_leq_k[None,:,:,None],
          axis=3)
        
        mk = tf.reduce_min(mk_j, axis=2)

        # We don't want to consider any margins that correspond to a k that
        # corresponds to a set of classes that do not match any of our affinity
        # sets.
        return tf.where(
            self._affinity_sets.where_matches(where_in_top_k),
            mk,
            -np.inf)
