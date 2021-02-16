# gloro
Library for training globally-robust neural networks.

# Installation

Simply install via pip:
```
pip install gloro
```

Alternatively, you can install from the source repository:

1. Clone the [repository](https://github.com/klasleino/gloro) and change into its root directory.

2. Install from source via
```
pip install -e .
```

# Usage

### Constructing GloRo Nets

The `GloroNet` class subclasses `tf.keras.models.Model`.
It can be built similarly to a keras model, except that it takes an additional parameter, `epsilon`, which specifies the robustness radius certified by the GloroNet. 
For example:
```python
from gloro import GloroNet


x = Input(5)
z = Dense(6)(x)
z = Activation('relu')(z)
z = Dense(7)(z)
z = Activation('relu')(z)
y = Dense(3)(z)

gloronet = GloroNet(x, y, epsilon=0.5)
```

A `GloroNet` can also be constructed from an existing model. 
The model is assumed to have *logit* outputs (i.e., there is no softmax at the last layer).
For example:
```python
from gloro import GloroNet


x = Input(5)
z = Dense(6)(x)
z = Activation('relu')(z)
z = Dense(7)(z)
z = Activation('relu')(z)
y = Dense(3)(z)

f = Model(x, y)

gloronet = GloroNet(model=f, epsilon=0.5)
```

### Training GloRo Nets

`GloroNet` models can be trained similarly to a standard Keras `Model` using the `fit` method.
The `training` package provides several useful modules for training GloRo Nets.
An example of training a `GloroNet` model is given below:
```python
from gloro.training.callbacks import EpsilonScheduler
from gloro.training.callbacks import UpdatePowerIterates
from gloro.training.losses import Crossentropy
from gloro.training.metrics import clean_acc
from gloro.training.metrics import vra


gloronet.compile(
    optimizer='adam',

    # This is the same as standard categorical crossentropy, except that (1) it
    # assumes it is operating on logits rather than probits, and (2) it adds an
    # extra column of zeros to the true labels.
    loss=Crossentropy(),
    
    metrics=[
        # As the name suggests, this calculates the clean accuracy of the
        # GloRo Net.
        clean_acc,

        # This is the same as Keras' 'acc' metric, but it is named to indicate
        # what it conceptually represents.
        vra,
    ])

gloronet.fit(
    X,
    Y,
    epochs=10,
    batch_size=16,
    callbacks=[
        # You should typically include this callback when training a `GloroNet`
        # model. This allows the model to maintain the state of the power
        # method iterates over training, and ensures that the computed 
        # Lipschitz constant has converged prior to the start of each epoch and
        # at the end of training. When calling `fit` on a `GloroNet`, this is
        # included by default, but it can also be added explicitly, like in
        # this example, or it can be disabled by passing 
        # `update_iterates=False` to `fit`.
        UpdatePowerIterates(),

        # It is often useful to begin with a small robustness radius and grow
        # it over time so that the GloRo Net learns to make accurate 
        # predictions in addition to robust ones. More detail on the schedule
        # options (e.g., 'logarithmic') is provided below.
        EpsilonScheduler('logarithmic')
    ])
```

As noted above, the `EpsilonScheduler` callback can be configured with a few different schedule options, listed below:

* `'fixed'`

  > This is essentially the same as having no schedule&mdash;epsilon will remain fixed during training.

* `'linear'` 

  > Epsilon is scaled linearly over the course of training from 0 to the value of epsilon initially set on the `GloroNet` object.

* `'linear_half'`

  > This is the same as `'linear'`, but the scaling takes place over the first half of training, and then epsilon remains fixed for the rest of training.

* `'logarithmic'`

  > Epsilon is increased over the course of training to reach the value of epsilon initially set on the `GloroNet` object by the end of training. As compared to `'linear'`, this schedule increases epsilon quickly at first, and decreases the rate of increase over time.

* `'logarithmic_half'`

  > this is the same as `'logarithmic'`, but the scaling takes place over the first half of training, and then epsilon remains fixed for the rest of training.

GloRo Nets can also be trained using TRADES loss.
The `Trades` loss function takes a parameter, `lam`, that represents the weight given to the robust part of the objective.
An example is shown below.
```python
from gloro.training.callbacks import TradesScheduler
from gloro.training.losses import Trades
from gloro.training.metrics import clean_acc
from gloro.training.metrics import vra


gloronet.compile(
    optimizer='adam',
    loss=Trades(2.),
    metrics=[
        clean_acc,
        vra,
    ])

gloronet.fit(
    X,
    Y,
    epochs=10,
    batch_size=16,
    callbacks=[
        # It is often useful to begin with a small TRADES parameter and
        # increase it over time so that the GloRo Net learns to make accurate 
        # predictions in addition to robust ones. The `TradesScheduler` 
        # callback admits the same set of schedule options as
        # `EpsilonScheduler`, described above.
        TradesScheduler('linear')
    ])
```

### More about the `GloroNet` Class

#### Saving and Loading
`GloroNet` models can be saved and loaded using the standard Keras model serialization API; however, to load a GloroNet, the `custom_objects` field has to be provided to Keras' `load_model` function.
In order to make this more convenient, a constant dictionary containing all custom objects in the `gloro` library is provided.
For example:
```python
from gloro import GLORO_CUSTOM_OBJECTS


gloronet.save('gloronet.h5')

loaded_gloronet = tf.keras.models.load_model(
    'gloronet.h5', custom_objects=GLORO_CUSTOM_OBJECTS)
```

#### `GloroNet` Properties and Methods

The `Gloronet` class provides some properties and methods that may be useful.
These properties are described below.

* `epsilon` 

  > The robustness radius certified by this GloRo Net. This property is settable, so it can be changed to certify a different robustness radius.

* `f`

  > The underlying model instrumented by the GloRo Net. This property is read-only.

* `lipschitz_constant(X)`
  
  > Gives the Lipschitz constant for each of the points in `X`. The value in the position of the predicted class of each point is `-1` to signify that this value should be ignored.

* `predict_clean(*args, **kwargs)`
  
  > Gets the predictions without the added bottom class.

* `freeze_lipschitz_constant()`

  > Converges the power-method iterates and then returns a new network where the Lipschitz constant is hard-coded rather than computed from the weights. The new model will make more efficient predictions, but it can no longer be trained.

* `refresh_iterates()`

  > Refreshes and converges the power-method iterates. This should be called before test-time certification. If the model was trained with the `UpdatePowerIterates` callback, this will have been called automatically at the end of training.
