import json
import numpy as np
import os
import shutil
import tarfile
import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.models import Model

import gloro

from gloro.lc import LipschitzComputationStrategy
from gloro.training.callbacks import UpdatePowerIterates
from gloro.training.losses import get as get_loss
from gloro.utils import get_value
from gloro.utils import set_value


class GloroNet(Model):
    """
    TODO: docstring.
    """
    def __init__(
        self,
        inputs=None,
        outputs=None,
        epsilon=None,
        *,
        model=None,
        _lc_frozen=False,
        _hardcoded_lc=1.,
        _skip_init=False,
        **kwargs,
    ):
        if _skip_init:
            # This can be used by subclasses of `GloroNet` that want to call
            # `super().__init__` to run `Model.__init__` but *not* this
            # initializer.
            super().__init__(inputs, outputs, **kwargs)
            return

        # Validate the provided parameters.
        if epsilon is None:
            raise ValueError('`epsilon` is required')

        if model is None and (inputs is None or outputs is None):
            raise ValueError(
                'must specify either `inputs` and `outputs` or `model`')

        if model is not None and inputs is not None and outputs is not None:
            raise ValueError(
                'cannot specify both `inputs` & `outputs` and `model`')

        # Collect and validate inputs and outputs.
        if inputs is None or outputs is None:
            inputs = model.inputs
            outputs = model.outputs

        else:
            model = Model(inputs, outputs)

        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]

        if len(outputs) > 1:
            raise ValueError(
                f'only models with a single output are supported, but got '
                f'{len(outputs)} outputs'
            )
        output = outputs[0]

        if len(output.shape) != 2:
            raise ValueError(
                f'output shape must have shape (None, m), where m is the '
                f'number of classes, but got {output.shape}'
            )
        num_classes = output.shape[1]

        if not hasattr(model.layers[-1], 'kernel'):
            raise ValueError(
                f'last layer is expected to have a `kernel` attribute, but no '
                f'such attribute was found on {model.layers[-1]}. It is '
                f'possible that the model you supplied ends with an activation '
                f'such as softmax, which is not accepted; in this case, you '
                f'should provide a model that ends with *logits* instead.'
            )

        super().__init__(inputs, outputs, **kwargs)

        self._epsilon = self._epsilon = tf.Variable(
            epsilon, dtype=K.floatx(), name='epsilon', trainable=False
        )
        self._lipschitz_computers = [
            LipschitzComputationStrategy.for_layer(layer)
            for layer in model.layers[:-1]
        ]
        self._num_classes = num_classes
        self._f = GloroNet._ModelContainer(model)

        # We allow the model to freeze its sub-Lipschitz constant after training
        # in order to facilitate even faster prediction and certification at
        # test time.
        self._lc_frozen = tf.Variable(
            int(_lc_frozen),
            name='lc_frozen',
            trainable=False,
        )
        self._hardcoded_lc = tf.Variable(
            _hardcoded_lc,
            name='hardcoded_lc',
            trainable=False,
        )


    @property
    def epsilon(self):
        return get_value(self._epsilon)

    @epsilon.setter
    def epsilon(self, new_value):
        set_value(self._epsilon, new_value)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def f(self):
        """
        This allows us to refer to the underlying model being "robustified" by
        the GloRo Net.
        """
        return self._f.model

    @property
    def lc_frozen(self):
        return bool(get_value(self._lc_frozen))

    # TODO remove this!!!
    @lc_frozen.setter
    def lc_frozen(self, new_value):
        set_value(self._lc_frozen, int(new_value))


    # Lipschitz computation.

    @property
    def sub_lipschitz(self):
        """
        This is an upper bound on the Lipschitz constant up to the penultimate
        layer.
        """
        return tf.switch_case(
            self._lc_frozen * 1,
            branch_fns={
                0: lambda:
                    tf.reduce_prod([
                        lipschitz() for lipschitz in self._lipschitz_computers
                    ]),
                1: lambda: self._hardcoded_lc * 1.,
            },
        )

    def _K_ij(self, k, j):
        """
        K_ij is the Lipschitz constant on the margin y_j - y_i.
        """
        kW = k * self.layers[-1].kernel

        # Get the weight column of the predicted class.
        kW_j = tf.gather(tf.transpose(kW), j)

        # Get weights that predict the value y_j - y_i for all i != j.
        kW_ij = kW_j[:,:,None] - kW[None]

        return tf.sqrt(tf.reduce_sum(kW_ij * kW_ij, axis=1))

    def lipschitz_constant(self, X=None, y=None):
        if y is None:
            if X is None:
                y = tf.eye(self.num_classes)
            else:
                y = self.f(X)

        # j is the predicted class, and y_j is the corresponding logit.
        j = tf.argmax(y, axis=1)
        y_j = tf.reduce_max(y, axis=1, keepdims=True)

        # K_ij is the Lipschitz constant on the margin y_j - y_i.
        K_ij = self._K_ij(self.sub_lipschitz, j)

        return tf.where(
            tf.equal(y, y_j),
            tf.zeros_like(K_ij) - 1.,
            K_ij,
        )

    def freeze_lc(self, threshold=1e-4, max_tries=100):
        # Make sure sub-Lipschitz constant has reached convergence.
        self.run_lc_to_convergence(threshold=threshold, max_tries=max_tries)

        # Once the Lipschitz constant is frozen, the model should not be trained
        # further unless it is unfrozen.
        self.f.trainable = False

        set_value(self._hardcoded_lc, self.sub_lipschitz)
        set_value(self._lc_frozen, True)

        return self

    def unfreeze_lc(self):
        self.f.trainable = True
        set_value(self._lc_frozen, False)
        return self

    def run_lc_to_convergence(self, threshold=1e-4, max_tries=100):

        def body(i, k_prev, diff):
            k = tf.stop_gradient(self.sub_lipschitz)
            return [i + 1, k, tf.abs(k_prev - k)]

        tf.while_loop(
            lambda i, k, diff: tf.logical_and(i < max_tries, diff > threshold),
            body,
            [tf.constant(0), self.sub_lipschitz, tf.constant(np.inf)]
        )
        return self


    # Prediction variations.

    def predict_clean(self, X, **kwargs):
        return self.f.predict(X, **kwargs)

    def certified_radius(self, X):
        return self._certified_radius(self.f(X)).numpy()

    def _certified_radius(self, y):
        # j is the predicted class, and y_j is the corresponding logit.
        j = tf.argmax(y, axis=1)
        y_j = tf.reduce_max(y, axis=1, keepdims=True)

        # K_ij is the Lipschitz constant on the margin y_j - y_i.
        K_ij = self._K_ij(self.sub_lipschitz, j)

        margins = y_j - y # shape: (None, m)

        # This is a certified lower bound on the distance from the point to the
        # boundary with class i, where i != j.
        radius_i = tf.where(
            tf.equal(y, y_j),
            np.infty + tf.zeros_like(y),
            margins / K_ij,
        )

        return tf.reduce_min(radius_i, axis=1)

    def predict_with_certified_radius(self, X):
        y_pred = self.f(X)
        radius = self._margin_layer.certified_radius(y_pred)

        return y_pred.numpy(), radius.numpy()


    # Overriding `keras.Model` functions.

    def call(self, X, training=False):
        y = self.f(X, training=training)

        k = self.sub_lipschitz
        W = self.layers[-1].kernel

        # j is the predicted class, and y_j is the corresponding logit.
        j = tf.argmax(y, axis=1)
        y_j = tf.reduce_max(y, axis=1, keepdims=True)

        # K_ij is the Lipschitz constant on the margin y_j - y_i.
        K_ij = self._K_ij(k, j)

        y_bot_i = y + self._epsilon * K_ij

        # `y_bot_i` will be zero at the position of class j. However, we don't
        # want to consider this class, so we replace the zero with negative
        # infinity so that when we find the maximum component for `y_bot_i` we
        # don't get zero as a result of all of the components we care aobut
        # being negative.
        y_bot_i = tf.where(
            tf.equal(y, y_j),
            -np.infty + tf.zeros_like(y_bot_i),
            y_bot_i,
        )

        # y_bot represents the amount the most competitive non-predicted class
        # can gain on the predicted class.
        y_bot = tf.reduce_max(y_bot_i, axis=1, keepdims=True)

        return tf.concat([y, y_bot], axis=1)


    # Keras models can pass loss functions and metrics to `compile` in the form
    # of strings, for common losses/metrics. Here, we override `compile` to
    # enable the same for the built in losses implemented in `gloro.losses`.
    def compile(self, **kwargs):
        if 'loss' in kwargs:
            if isinstance(kwargs['loss'], str):
                kwargs['loss'] = get_loss(kwargs['loss'])

        if 'metrics' in kwargs:
            metrics = kwargs['metrics']
            new_metrics = []

            for metric in metrics:
                if metric in gloro.constants.GLORO_CUSTOM_OBJECTS:
                    metric = gloro.constants.GLORO_CUSTOM_OBJECTS[metric]

                new_metrics.append(metric)

            kwargs['metrics'] = new_metrics

        super().compile(**kwargs)

    # When training a GloRo Net, we typically want to (1) refresh the power
    # iterates at the beginning of each epoch and run them to convergence, and
    # (2) converge the power iterates precisely at the end of training to ensure
    # a sound upper bound. Here, we override `fit` to add a callback to do this
    # by default, so the devoloper doesn't have to remember to do this manually.
    def fit(self, *args, update_iterates=True, **kwargs):
        # If we are going to train, the Lipschitz constant should be unfrozen.
        self.unfreeze_lc()

        if update_iterates:
            includes_update_iterates = False

            if 'callbacks' in kwargs:
                callbacks = kwargs['callbacks']

                for callback in callbacks:
                    if isinstance(callback, UpdatePowerIterates):
                        includes_update_iterates = True

            else:
                callbacks = []

            if not includes_update_iterates:
                kwargs['callbacks'] = [
                    UpdatePowerIterates(verbose=True),
                    *callbacks
                ]

        return super().fit(*args, **kwargs)

    # We save and load GloRo Nets using our own file format.
    def save(self, file_name, overwrite=True):
        if file_name.endswith('.h5'):
            file_name = file_name[:-3]

        elif file_name.endswith('.gloronet'):
            file_name = file_name[:-9]

        try:
            os.mkdir(file_name)

            self.f.save(f'{file_name}/f.h5')

            with open(f'{file_name}/config.json', 'w') as json_file:
                json.dump(self.get_config(), json_file)

            with tarfile.open(f'{file_name}.gloronet', 'w:gz') as tar:
                tar.add(file_name, arcname=os.path.basename(file_name))

        finally:
            shutil.rmtree(file_name)

    def get_config(self):
        return {
            'epsilon': float(self.epsilon),
            '_lc_frozen': bool(self.lc_frozen),
            '_hardcoded_lc': float(get_value(self._hardcoded_lc)),
        }

    @classmethod
    def load_model(cls, file_name, custom_objects={}, converge=True):
        custom_objects = dict(
            gloro.constants.GLORO_CUSTOM_OBJECTS.copy(), **custom_objects
        )

        if file_name.endswith('.h5'):
            file_name = file_name[:-3]

        elif file_name.endswith('.gloronet'):
            file_name = file_name[:-9]

        temp_dir = f'{file_name}___'

        try:
            os.mkdir(temp_dir)

            with tarfile.open(f'{file_name}.gloronet', 'r:gz') as tar:

                for member in tar:
                    if member.name.endswith('.h5'):
                        tar.extract(member, f'{temp_dir}')

                        model = tf.keras.models.load_model(
                            f'{temp_dir}/{member.name}',
                            custom_objects=custom_objects
                        )

                    elif member.name.endswith('.json'):
                        with tar.extractfile(member) as f:
                            config = json.load(f)
        finally:
            shutil.rmtree(temp_dir)

        if '_lc_frozen' not in config or not config['_lc_frozen']:
            if converge:
                return cls(model=model, **config).run_lc_to_convergence()
        else:
            return cls(model=model, **config)


    # This allows us to store the original model on the `GloroNet` instance
    # without keras finding it.
    class _ModelContainer(object):
        def __init__(self, model):
            self.model = model
