import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate

import gloro

from gloro.models import GloroNet
from gloro.relaxations.affinity_sets import AffinitySet
from gloro.relaxations.layers import LipschitzMarginAffinity
from gloro.relaxations.layers import LipschitzMarginRtk
from gloro.training.callbacks import UpdatePowerIterates
from gloro.training.losses import get as get_loss


class RtkGloroNet(GloroNet):
    def __init__(
        self, 
        inputs=None, 
        outputs=None, 
        epsilon=None, 
        k=None,
        num_iterations=5,
        model=None, 
        _skip_init=False,
        **kwargs,
    ):
        if _skip_init:
            # We have to do this because keras requires __init__ to be called on
            # *all* subclasses.
            super().__init__(inputs, outputs, _skip_init=True, **kwargs)
            return

        if epsilon is None:
            raise ValueError('`epsilon` is required')

        if k is None:
            raise ValueError('`k` is required')
        
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

        margin_layer = LipschitzMarginRtk(
            epsilon, k, model, num_iterations=num_iterations)

        margin = margin_layer(*outputs)

        margin, k_guarantee, loosest_k = margin

        margin = [
            Activation('linear', name='pred')(margin),
            Activation('linear', name='guarantee')(k_guarantee),
            Concatenate(name='pred_top_k')(
                [margin, loosest_k])
        ]
        
        super().__init__(inputs, margin, None, _skip_init=True, **kwargs)

        self._inputs = inputs
        self._output = outputs[0]
        self._margin_layer = margin_layer
        self._f = GloroNet._ModelContainer(model)

    @property
    def k(self):
        return self._margin_layer.k

    @k.setter
    def k(self, new_k):
        self._margin_layer.k = new_k

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


    # Overriding `keras.Model` functions.

    def predict(self, *args, **kwargs):
        return super().predict(*args, **kwargs)[0]

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
        return {
            'epsilon': float(self.epsilon),
            'k': int(self.k),
            'num_iterations': int(self.num_iterations),
        }


class AffinityGloroNet(RtkGloroNet):
    def __init__(
        self,
        inputs=None,
        outputs=None,
        epsilon=None,
        affinity_sets=None,
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

        if affinity_sets is None:
            raise ValueError('`affinity_sets` is required')

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

        if affinity_sets.mask.shape[1] != model.output_shape[1]:
            raise ValueError(
                f'affinity sets defined for a number of classes that does not '
                f'match the output shape of the model: '
                f'{affinity_sets.mask.shape[1]} vs {model.output_shape[1]}')

        k = affinity_sets.max_set_size

        margin_layer = LipschitzMarginAffinity(
            epsilon, k, affinity_sets, model, num_iterations=num_iterations)

        margin = margin_layer(*outputs)

        margin, k_guarantee, loosest_k = margin

        margin = [
            Activation('linear', name='pred')(margin),
            Activation('linear', name='guarantee')(k_guarantee),
            Concatenate(name='pred_top_k')(
                [margin, loosest_k])
        ]

        super().__init__(inputs, margin, None, None, _skip_init=True, **kwargs)

        self._inputs = inputs
        self._output = outputs[0]
        self._margin_layer = margin_layer
        self._f = GloroNet._ModelContainer(model)

    @property
    def affinity_sets(self):
        return self._margin_layer.affinity_sets

    def get_config(self):
        return {
            'epsilon': float(self.epsilon),
            'affinity_sets': str(self.affinity_sets),
            'num_iterations': int(self.num_iterations),
        }
