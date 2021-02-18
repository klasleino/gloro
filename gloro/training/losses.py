import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.losses import Loss
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from gloro.training.utils import add_extra_column


class Crossentropy(Loss):

    def __init__(self, sparse=False, **kwargs):
        super().__init__()

        kwargs['from_logits'] = True

        if sparse:
            self._loss = SparseCategoricalCrossentropy(**kwargs)
        else:
            self._loss = CategoricalCrossentropy(**kwargs)

        self._sparse = sparse

    def call(self, y_true, y_pred):
        # Add an extra column of zeros for the bottom class.
        if not self._sparse:
            y_true = add_extra_column(y_true)

        return self._loss.call(y_true, y_pred)

    def get_config(self):
        config = self._loss.get_config()
        config.update({
            'sparse': self._sparse
        })
        return config


class Trades(Loss):

    def __init__(self, lam=1., sparse=False, robust_loss_string=None):
        super().__init__()

        self._lam = tf.Variable(lam, trainable=False)
        K.set_value(self._lam, lam)

        self._robust_loss_string = robust_loss_string

        if sparse:
            self._crossentropy = SparseCategoricalCrossentropy(from_logits=True)
        else:
            self._crossentropy = CategoricalCrossentropy(from_logits=True)

        if robust_loss_string is None:
            self._robust_loss = CategoricalCrossentropy(from_logits=False)

        else:
            self._robust_loss = tf.losses.get(robust_loss_string)

        self._sparse = sparse

    @property
    def lam(self):
        return K.get_value(self._lam)

    @lam.setter
    def lam(self, new_lam):
        K.set_value(self._lam, new_lam)

    def call(self, y_true, y_pred):
        # Add an extra column of zeros for the bottom class.
        if not self._sparse:
            y_true = add_extra_column(y_true)

        # Encourage predicting the correct class, even non-robustly.
        if self._sparse:
            standard_loss = self._crossentropy(y_true, y_pred[:, :-1])
        else:
            standard_loss = self._crossentropy(y_true[:, :-1], y_pred[:, :-1])

        # Encourage predicting robustly, even incorrectly. We take the robust
        # loss but using the model's prediction as the ground truth.
        new_ground_truth = add_extra_column(tf.nn.softmax(y_pred[:, :-1]))

        robust_loss = self._robust_loss(new_ground_truth, tf.nn.softmax(y_pred))

        return standard_loss + self._lam * robust_loss

    def get_config(self):
        return {
            'lam': self.lam,
            'robust_loss_string': self._robust_loss_string,
            'sparse': self._sparse
        }
