import tensorflow.keras.backend as K

from tensorflow.keras.models import Model

from gloro.layers import LipschitzMargin
from gloro.training.callbacks import UpdatePowerIterates


class GloroNet(Model):
    def __init__(
            self, 
            inputs=None, 
            outputs=None, 
            epsilon=None, 
            num_iterations=5,
            maintain_state=True,
            model=None, 
            hardcoded_kW=None, 
            **kwargs):

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
            epsilon, 
            model, 
            num_iterations=num_iterations, 
            maintain_state=maintain_state,
            hardcoded_kW=hardcoded_kW)

        margin = margin_layer(*outputs)
        
        super().__init__(inputs, margin, **kwargs)
        
        self._inputs = inputs
        self._outputs = outputs
        
        self._margin_layer = margin_layer
        self._f = GloroNet.__ModelContainer(model)

        self._lipschitz_constant_fn = K.function(
            self._inputs,
            [self._margin_layer._lipschitz_constant_tensor])

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
    def maintain_state(self):
        return self._margin_layer.maintain_state

    @property
    def f(self):
        return self._f.model

    def lipschitz_constant(self, X):
        if not isinstance(X, (list, tuple)):
            X = [X]

        return self._lipschitz_constant_fn(X)[0]

    def predict_clean(self, *args, **kwargs):
        return self._f.model.predict(*args, **kwargs)

    def freeze_lipschitz_constant(self, converge=True):
        if converge:
            self.refresh_iterates(batch_size=100)

        frozen = GloroNet(
            epsilon=self.epsilon,
            model=self._f.model,
            num_iterations=0,
            hardcoded_kW=K.get_value(self._margin_layer._kW))

        frozen.trainable = False

        return frozen

    def refresh_iterates(
            self, 
            converge=True, 
            convergence_threshold=1e-4, 
            max_tries=1000, 
            batch_size=None,
            verbose=False):

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
                    UpdatePowerIterates(verbose=False),
                    *callbacks
                ]

        return super().fit(*args, **kwargs)
        
    def get_config(self):
        config = super().get_config()
        config.update({
            'epsilon': self.epsilon,
            'model': self._f.model.get_config()
        })
        return config
      
    @classmethod
    def from_config(cls, config):
        model = Model.from_config(config['model'])
        
        return GloroNet(model=model, epsilon=config['epsilon'])
        
    class __ModelContainer(object):
        def __init__(self, model):
            self.model = model
