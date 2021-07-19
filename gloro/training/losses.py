import numpy as np
import tensorflow as tf

from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.losses import KLDivergence
from tensorflow.keras.losses import Loss
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from gloro.utils import add_extra_column
from gloro.utils import get_value
from gloro.utils import set_value


class Crossentropy(Loss):

    def __init__(self, sparse=False, temperature=1., **kwargs):
        super().__init__(**kwargs)

        kwargs['from_logits'] = True

        if sparse:
            self._loss = SparseCategoricalCrossentropy(**kwargs)
        else:
            self._loss = CategoricalCrossentropy(**kwargs)

        self._sparse = sparse

        self._temperature = tf.Variable(temperature, trainable=False)

    @property
    def temperature(self):
        return get_value(self._temperature)

    @temperature.setter
    def temperature(self, new_temperature):
        set_value(self._temperature, new_temperature)

    def call(self, y_true, y_pred):
        # Add an extra column of zeros for the bottom class.
        if not self._sparse:
            y_true = add_extra_column(y_true)

        return self._loss.call(y_true, y_pred / self._temperature)

    def get_config(self):
        config = self._loss.get_config()
        config.update({
            'sparse': self._sparse,
            'temperature': self.temperature,
        })
        return config


class Trades(Loss):

    def __init__(
        self, 
        lam=1., 
        sparse=False, 
        robust_loss_string='kl_divergence', 
        temperature=1.,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._lam = tf.Variable(lam, trainable=False)

        self._robust_loss_string = robust_loss_string

        if 'from_logits' in kwargs:
            del kwargs['from_logits']

        if sparse:
            self._crossentropy = SparseCategoricalCrossentropy(
                from_logits=True, **kwargs)
        else:
            self._crossentropy = CategoricalCrossentropy(
                from_logits=True, **kwargs)

        if robust_loss_string is None or robust_loss_string == 'crossentropy':
            self._robust_loss = CategoricalCrossentropy(
                from_logits=False, **kwargs)

        elif robust_loss_string == 'kl_divergence':
            self._robust_loss = KLDivergence(**kwargs)

        else:
            raise ValueError(f'unsupported robust loss: {robust_loss_string}')

        self._sparse = sparse

        self._temperature = tf.Variable(temperature, trainable=False)

    @property
    def lam(self):
        return get_value(self._lam)

    @lam.setter
    def lam(self, new_lam):
        set_value(self._lam, new_lam)

    @property
    def temperature(self):
        return get_value(self._temperature)

    @temperature.setter
    def temperature(self, new_temperature):
        set_value(self._temperature, new_temperature)

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
        y_pred_soft = tf.nn.softmax(y_pred)

        new_ground_truth = add_extra_column(tf.nn.softmax(y_pred[:, :-1]))

        robust_loss = self._robust_loss(
            new_ground_truth, y_pred_soft / self._temperature)

        # Combine the standard and robust terms.
        return standard_loss + self._lam * robust_loss

    def get_config(self):
        return {
            'lam': self.lam,
            'sparse': self._sparse,
            'robust_loss_string': self._robust_loss_string,
            'temperature': self.temperature,
        }


def get(loss):
    sparse = False
    if loss.startswith('sparse_'):
        loss = loss.split('sparse_')[1]
        sparse = True

    if loss == 'crossentropy':
        return Crossentropy(sparse=sparse)

    elif loss.startswith('trades_ce'):
        if loss.startswith('trades_ce.'):
            return Trades(
                float(loss.split('trades_ce.')[1]), 
                robust_loss_string='crossentropy',
                sparse=sparse)

        else:
            return Trades(robust_loss_string='crossentropy', sparse=sparse)

    elif loss.startswith('trades_kl'):
        if loss.startswith('trades_kl.'):
            return Trades(
                float(loss.split('trades_kl.')[1]), 
                robust_loss_string='kl_divergence',
                sparse=sparse)
        else:
            return Trades(robust_loss_string='kl_divergence', sparse=sparse)

    else:
        raise ValueError(f'unknown loss function: {loss}')
