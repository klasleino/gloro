import tensorflow as tf


def clean_acc(y_true, y_pred):
    return tf.reduce_mean(tf.cast(
        tf.equal(tf.argmax(y_true, axis=1), tf.argmax(y_pred[:, :-1], axis=1)),
        'float32'))

def vra(y_true, y_pred):
    return tf.reduce_mean(tf.cast(
        tf.equal(tf.argmax(y_true, axis=1), tf.argmax(y_pred, axis=1)),
        'float32'))
