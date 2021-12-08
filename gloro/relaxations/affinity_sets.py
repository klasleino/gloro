import numpy as np
import re
import tensorflow as tf

from gloro.utils import get_value


class AffinitySet(object):
    def __init__(self, one_hot_encoding):
        self._mask = tf.constant(one_hot_encoding, dtype='int32')
        self._num_classes = one_hot_encoding.shape[1]

        self._max_set_size = tf.reduce_max(
            tf.reduce_sum(self._mask, axis=1))

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def mask(self):
        return get_value(self._mask)

    @property
    def max_set_size(self):
        return get_value(self._max_set_size)

    def matches(self, top_k_preds):
        asm = self._mask[None,None]
        ccm = self._get_cumulative_class_map(top_k_preds)[:,:,None]

        return tf.reduce_prod(1 - ccm + asm * ccm, axis=3)

    def where_matches(self, top_k_preds):
        asm = self._mask[None,None]
        ccm = self._get_cumulative_class_map(top_k_preds)[:,:,None]

        # ASM : ( -- , -, n, c)
        # CCM : (None, k, -, c)

        # We take the sum (capped at a value of 1) as a logical OR over all 
        # affinity sets because we only care whether a given set of top-k 
        # classes matches with any affinity set.
        where_matches = tf.reduce_sum(
            # We take the product (i.e., a logical AND) over all class positions
            # since the cumulative class mask must be a match for the affinity 
            # set mask for all class positions.
            tf.reduce_prod(
                # This represents the logical formula, CCM_i ==> ASM_i. In other
                # words, the cumulative class mask should only be 1 where the
                # affinity set mask is 1, in order for the corresponding k to be
                # considered a match for that affinity set.
                1 - ccm + asm * ccm, 
                axis=3),
            axis=2) > 0

        return tf.cast(where_matches, 'bool')

    def _get_cumulative_class_map(self, top_k_preds):
        # Assumes `top_k_preds` has shape (None, k, c), where c is the number of
        # classes, and that it represents a one-hot encoding of the kth-highest
        # class for each k.
        return tf.math.cumsum(tf.cast(top_k_preds, 'int32'), axis=1)

    def as_indices(self):
        return [
            [
                int(i) for i, keep in enumerate(affinity_set) if keep
            ]
            for affinity_set in self._mask
        ]

    def __str__(self):
        return str(self.as_indices()).replace(' ', '')[1:-1]

    @staticmethod
    def from_string(s):
        # Expects strings of the form like the following example:
        #
        #   '[0][1,2,3][3,4]'
        #
        return AffinitySet.from_class_indices([
            [int(e) for e in block[1:-1].split(',')]
            for block in re.findall(r'\[.*?\]', s)
        ])

    @staticmethod
    def from_class_indices(class_indices):
        num_classes = max([max(s) for s in class_indices]) + 1

        one_hot_encoding = np.array([
            (np.arange(num_classes)[:,None] == np.array(s)[None]).sum(axis=1)
            for s in class_indices
        ])

        return AffinitySet(one_hot_encoding)

    @staticmethod
    def from_one_hot(one_hot_encoding):
        return AffinitySet(one_hot_encoding)

    def __str__(self):
        sets = '\n'.join([
            f'  {str([i for i, v in enumerate(s) if v == 1])}'
            for s in self._mask.numpy()
        ])
        return f'AffinitySet: [\n{sets}]'

    def __repr__(self):
        return str(self)
