if __name__ == '__main__':
    import os

    # Stop tensorflow's obnoxious logging.
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json
import tensorflow as tf
import tensorflow_datasets as tfds

from scriptify import scriptify
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.optimizers import SGD

from gloro import GloroNet
from gloro.training import losses
from gloro.training.callbacks import EpsilonScheduler 
from gloro.training.callbacks import LrScheduler
from gloro.training.callbacks import TradesScheduler
from gloro.training.metrics import clean_acc
from gloro.training.metrics import vra
from gloro.training.metrics import rejection_rate
from gloro.utils import print_if_verbose

import architectures
from utils import get_data
from utils import get_optimizer


def train_gloro(
    dataset,
    architecture,
    epsilon,
    power_iterations=5,
    epsilon_schedule='fixed',
    loss='crossentropy',
    augmentation='standard',
    epochs=None,
    batch_size=None,
    optimizer='adam',
    lr=1e-3,
    lr_schedule='fixed',
    trades_schedule=None,
    verbose=True,
):
    _print = print_if_verbose(verbose)

    # Load data and set up data pipeline.
    _print('loading data...')

    train, test, metadata = get_data(dataset, batch_size, augmentation)

    input_shape = metadata.features['image'].shape
    num_classes = metadata.features['label'].num_classes

    # Create the model.
    _print('creating model...')

    try:
        _orig_architecture = architecture
        params = '{}'

        if '.' in architecture:
            architecture, params = architecture.split('.', 1)

        architecture = getattr(architectures, architecture)(
            input_shape, num_classes, **json.loads(params))

    except:
        raise ValueError(f'unknown architecture: {_orig_architecture}')

    g = GloroNet(*architecture, epsilon, num_iterations=power_iterations)

    if verbose:
        g.summary()

    # Compile and train the model.
    _print('compiling model...')

    g.compile(
        loss=losses.get(loss),
        optimizer=get_optimizer(optimizer, lr), 
        metrics=[clean_acc, vra, rejection_rate])

    g.fit(
        train,
        epochs=epochs, 
        validation_data=test,
        callbacks=[
            EpsilonScheduler(epsilon_schedule),
            LrScheduler(lr_schedule),
        ] + ([TradesScheduler(trades_schedule)] if trades_schedule else []))

    return g



if __name__ == '__main__':

    @scriptify
    def script(
        dataset,
        architecture,
        epsilon,
        power_iterations=5,
        epsilon_schedule='fixed',
        loss='crossentropy',
        augmentation='standard',
        epochs=100,
        batch_size=128,
        optimizer='adam',
        lr=1e-3,
        lr_schedule='decay_to_0.000001',
        trades_schedule=None,
        gpu=0,
    ):
        # Select the GPU and allow memory growth to avoid taking all the RAM.
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_visible_devices(gpus[gpu], 'GPU')
        device = gpus[gpu]

        for device in tf.config.experimental.get_visible_devices('GPU'):
            tf.config.experimental.set_memory_growth(device, True)

        g = train_gloro(
            dataset,
            architecture,
            epsilon,
            power_iterations=power_iterations,
            epsilon_schedule=epsilon_schedule,
            loss=loss,
            augmentation=augmentation,
            epochs=epochs,
            batch_size=batch_size,
            optimizer=optimizer,
            lr=lr,
            lr_schedule=lr_schedule,
            trades_schedule=trades_schedule)

        # Evaluate the model.
        train, test, metadata = get_data(dataset, batch_size, augmentation)

        train_eval = g.evaluate(train)
        test_eval = g.evaluate(test)

        results = {}

        results.update({
            'test_' + metric.name.split('pred_')[-1]: round(value, 4)
            for metric, value in zip(g.metrics, test_eval)
        })
        results.update({
            'train_' + metric.name.split('pred_')[-1]: round(value, 4)
            for metric, value in zip(g.metrics, train_eval)
        })

        print(results)

        return results
