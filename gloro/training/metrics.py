import tensorflow as tf


def clean_acc_sparse(y_true, y_pred):
    labels = tf.cast(y_true, 'int32')[:,0]

    return tf.reduce_mean(tf.cast(
        tf.equal(labels, tf.cast(tf.argmax(y_pred[:, :-1], axis=1), 'int32')),
        'float32'))

def clean_acc_cat(y_true, y_pred):
    labels = tf.argmax(y_true, axis=1, output_type='int32')[:,None]

    return clean_acc_sparse(labels, y_pred)

def clean_acc(y_true, y_pred):
    return tf.case([
        (tf.equal(tf.shape(y_true)[1], 1), 
            lambda: clean_acc_sparse(y_true, y_pred))], 
        default=lambda: clean_acc_cat(y_true, y_pred))


def vra_sparse(y_true, y_pred):
    labels = tf.cast(y_true, 'int32')[:,0]

    return tf.reduce_mean(tf.cast(
        tf.equal(labels, tf.cast(tf.argmax(y_pred, axis=1), 'int32')),
        'float32'))

def vra_cat(y_true, y_pred):
    labels = tf.argmax(y_true, axis=1, output_type='int32')[:,None]

    return vra_sparse(labels, y_pred)

def vra(y_true, y_pred):
    return tf.case([
        (tf.equal(tf.shape(y_true)[1], 1), 
            lambda: vra_sparse(y_true, y_pred))], 
        default=lambda: vra_cat(y_true, y_pred))


def rejection_rate(y_true, y_pred):
    bot_index = tf.cast(tf.shape(y_pred)[1] - 1, 'int64')

    return tf.reduce_mean(tf.cast(
        tf.argmax(y_pred, axis=1) == bot_index, 'float32'))
