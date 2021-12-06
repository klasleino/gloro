# gloro
Library for training provably-robust neural networks.

This repository implements the method described in this [paper](https://arxiv.org/pdf/2102.08452.pdf) (appearing in ICML 2021), and is maintained by the authors, Klas Leino, Zifan Wang, and Matt Fredrikson. If you use this code, please use the following citation:
```bibtex
@INPROCEEDINGS{leino21gloro,
    title = {Globally-Robust Neural Networks},
    author = {Klas Leino and Zifan Wang and Matt Fredrikson},
    booktitle = {International Conference on Machine Learning (ICML)},
    year = {2021},
}
```

The `gloro/relaxations` directory implements the extensions described in this [paper](https://arxiv.org/pdf/2106.06624.pdf) (appearing in NIPS 2021). If you use code from this directory, please additionally cite:
```bibtex
@INPROCEEDINGS{leino2021relaxing,
  title = {Relaxing Local Robustness},
  author = {Klas Leino and Matt Fredrikson},
  booktitle = {Neural Information Processing Systems (NIPS)},
  year = {2021}
}
```

# Best Results

For quick reference, we include our current best VRA (verified-robust accuracy) results here (these are more up-to-date and may surpass the results reported in the original paper). These results currently represent the state-of-the-art for deterministic L2 robustness certification. See `tools` for scripts to reproduce these results.

dataset       | radius | architecture | VRA
--------------|--------|--------------|------
MNIST         | 0.3    | 2C2F         | 0.957
MNIST         | 1.58   | 4C3F         | 0.628
CIFAR-10      | 0.141  | 6C2F         | 0.600
Tiny-Imagenet | 0.141  | 8C2F         | 0.224


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

### Examples

Example training scripts can be found in the `training` directory under [`tools`](https://github.com/klasleino/gloro/tree/master/tools)

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
The `gloro.training` package provides several useful modules for training GloRo Nets.
An example of training a `GloroNet` model is given below:
```python
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
    ])
```

The `compile` method is set up to also accept string shorthands for the loss and metrics, so the above code can be written more concisely and with fewer imports:
```python
gloronet.compile(
    optimizer='adam',
    loss='crossentropy',
    metrics=['clean_acc', 'vra'])

gloronet.fit(X, Y, epochs=10, batch_size=16)
```
See `gloro.training.losses.get` for the available loss shorthands.

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
        # predictions in addition to robust ones.
        TradesScheduler('linear')
    ])
```

#### Scheduler Callbacks
The `gloro.training` package also provides several useful callbacks for scheduling the learning rate, TRADES parameter, etc., during training.
See `gloro.training.callbacks` for the available scheduling callbacks, and `gloro.training.schedules` for the available schedule shorthands.

### More about the `GloroNet` Class

#### Saving and Loading
`GloroNet` models can be saved using the standard Keras model serialization API; however, to load a GloroNet, `GloroNet.load_model` should be used instead of `keras.models.load_model`. 
For example:
```python
from gloro import GloroNet

# The `gloro` library saves models with a '.gloronet' extension. This file
# contains the underlying model instrumented by the GloRo Net, as well as
# metadata associated with the `GloroNet` object.
gloronet.save('my_model.gloronet')

loaded_gloronet = GloroNet.load_model('my_model.gloronet')
```

#### `GloroNet` Properties and Methods

The `Gloronet` class provides some properties and methods that may be useful.
These properties are described below.

* `epsilon` 

  > The robustness radius certified by this GloRo Net. This property is settable, so it can be changed to certify a different robustness radius.

* `f`

  > The underlying model instrumented by the GloRo Net. This property is read-only.

* `lipschitz_constant()`
  
  > Gives the Lipschitz constant of for each pair of classes. The value in the diagonal is `-1` to signify that this value should be ignored.

* `predict_clean(*args, **kwargs)`
  
  > Gets the predictions without the added bottom class.

* `freeze_lipschitz_constant()`

  > Converges the power-method iterates and then hard-codes the Lipschitz constant such that it no longer needs to be computed from the model parameters. The frozen model will make more efficient predictions, but it can no longer be trained.

* `refresh_iterates()`

  > Refreshes and converges the power-method iterates. This should be called before test-time certification. If the model was trained with the `UpdatePowerIterates` callback, this will have been called automatically at the end of training.

# Main Contributers

* Klas Leino
* Zifan Wang
* Matt Fredrikson
