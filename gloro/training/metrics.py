import tensorflow as tf


def clean_acc(y_true, y_pred):
    labels = tf.case(
        [(tf.equal(tf.shape(y_true)[1], 1), 
            lambda: tf.cast(y_true, 'int32')[:,0])],
        lambda: tf.cast(tf.argmax(y_true, axis=1), 'int32'))

    return tf.reduce_mean(tf.cast(
        tf.equal(labels, tf.cast(tf.argmax(y_pred[:, :-1], axis=1), 'int32')),
        'float32'))


def vra(y_true, y_pred):
    labels = tf.case(
        [(tf.equal(tf.shape(y_true)[1], 1), 
            lambda: tf.cast(y_true, 'int32')[:,0])],
        lambda: tf.cast(tf.argmax(y_true, axis=1), 'int32'))

    return tf.reduce_mean(tf.cast(
        tf.equal(labels, tf.cast(tf.argmax(y_pred, axis=1), 'int32')),
        'float32'))
