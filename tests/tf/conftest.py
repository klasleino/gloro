import tensorflow as tf


def pytest_addoption(parser):
    parser.addoption('--gpu', type=int, action='store')

def pytest_configure(config):
    gpu = config.getoption('--gpu')

    if gpu is not None:
        device = tf.config.experimental.list_physical_devices('GPU')[gpu]

        tf.config.experimental.set_visible_devices(device, 'GPU')
        tf.config.experimental.set_memory_growth(device, True)
