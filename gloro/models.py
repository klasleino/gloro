import json
import os
import shutil
import tarfile
import tensorflow as tf

from tensorflow.keras.models import Model

import gloro

from gloro.layers.margin_layers import LipschitzMargin
from gloro.training.callbacks import UpdatePowerIterates
from gloro.training.losses import get as get_loss


class GloroNet(Model):
    def __init__(
        self, 
        inputs=None, 
        outputs=None, 
        epsilon=None, 
        num_iterations=5,
        model=None, 
        _skip_init=False,
        **kwargs,
    ):
        if _skip_init:
            # We have to do this because keras requires __init__ to be called on
            # *all* subclasses.
            super().__init__(inputs, outputs, **kwargs)
            return

        if epsilon is None:
            raise ValueError('`epsilon` is required')
        
        if model is None and (inputs is None or outputs is None):
            raise ValueError(
                'must specify either `inputs` and `outputs` or `model`')
            
        if model is not None and inputs is not None and outputs is not None:
            raise ValueError(
                'cannot specify both `inputs` and `outputs` and `model`')

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
                f'{len(outputs)} outputs')

        margin_layer = LipschitzMargin(
            epsilon, model, num_iterations=num_iterations)

        margin = margin_layer(*outputs)
        
        super().__init__(inputs, margin, **kwargs)

        self._inputs = inputs
        self._output = outputs[0]
        self._margin_layer = margin_layer
        self._f = GloroNet._ModelContainer(model)

    @property
    def epsilon(self):
        return self._margin_layer.epsilon

    @epsilon.setter
    def epsilon(self, new_epsilon):
        self._margin_layer.epsilon = new_epsilon

    @property
    def num_iterations(self):
        return self._margin_layer.num_iterations

    @num_iterations.setter
    def num_iterations(self, new_num_iterations):
        self._margin_layer.num_iterations = new_num_iterations

    @property
    def f(self):
        return self._f.model


    def lipschitz_constant(self, X=None, y=None):
        if X is None and y is None:
            return self._margin_layer.lipschitz_constant(
                tf.eye(self._output.shape[1]))

        elif X is not None:
            if not isinstance(X, (list, tuple)):
                X = [X]

            return self._margin_layer.lipschitz_constant(self.f(X))

        else:
            return self._margin_layer.lipschitz_constant(y)

    def certified_radius(self, x):
        return self._margin_layer.certified_radius(self.f(x))

    def predict_clean(self, *args, **kwargs):
        return self._f.model.predict(*args, **kwargs)

    def predict_with_certified_radius(self, x):
        y_pred = self.f(x)
        radius = self._margin_layer.certified_radius(y_pred)

        return y_pred.numpy(), radius.numpy()

    def freeze_lipschitz_constant(
        self, 
        converge=True, 
        convergence_threshold=1e-4, 
        max_tries=1000, 
        batch_size=None,
        verbose=False,
    ):
        if converge:
            self.refresh_iterates(
                converge=converge,
                convergence_threshold=convergence_threshold,
                max_tries=max_tries,
                batch_size=batch_size,
                verbose=verbose)

        self._margin_layer.freeze()
        self.trainable = False

        return self

    def refresh_iterates(
        self, 
        converge=True, 
        convergence_threshold=1e-4, 
        max_tries=1000, 
        batch_size=None,
        verbose=False,
    ):
        self._margin_layer.refresh_iterates(
            converge=converge,
            convergence_threshold=convergence_threshold,
            max_tries=max_tries,
            batch_size=batch_size,
            verbose=verbose)

        return self

    def update_iterates(self):
        self._margin_layer.update_iterates()

        return self


    # Overriding `keras.Model` functions.

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

    def fit(self, *args, update_iterates=True, **kwargs):
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

    @classmethod
    def load_model(cls, file_name, custom_objects={}, converge=True):
        custom_objects = dict(gloro.constants.GLORO_CUSTOM_OBJECTS.copy(), **custom_objects)

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
                            custom_objects=custom_objects)

                    elif member.name.endswith('.json'):
                        with tar.extractfile(member) as f:
                            config = json.load(f)
        finally:
            shutil.rmtree(temp_dir)

        if converge:
            return cls(model=model, **config).refresh_iterates()
        else:
            return cls(model=model, **config)

    def get_config(self):
        return {
            'epsilon': float(self.epsilon),
            'num_iterations': int(self.num_iterations),
        }


    class _ModelContainer(object):
        def __init__(self, model):
            self.model = model
