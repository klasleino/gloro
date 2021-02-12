import numpy as np
import tensorflow as tf


def add_extra_column(y):
    if isinstance(y, np.ndarray):
        return add_extra_column_np(y)
    else:
        return add_extra_column_tf(y)

def add_extra_column_tf(y):
    return tf.concat((y, tf.zeros((tf.shape(y)[0], 1))), axis=1)

def add_extra_column_np(y):
    return np.concatenate((y, np.zeros((y.shape[0], 1))), axis=1)
