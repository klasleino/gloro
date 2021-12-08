import tensorflow as tf


def rtk_vra_sparse(y_true, y_pred):
    labels = tf.cast(y_true, 'int32')[:,0]

    bot_index = tf.shape(y_pred)[1] - 2

    guarantee = tf.cast(y_pred[:, -1], 'int32')

    k = tf.reduce_max(guarantee)

    y_j, _ = tf.math.top_k(y_pred[:,:-2], k=k)

    pred_at_ground_truth = tf.gather(y_pred, labels, batch_dims=1)

    minimum_pred = tf.gather(y_j, guarantee - 1, batch_dims=1)

    return tf.reduce_mean(
        tf.cast(pred_at_ground_truth >= minimum_pred, 'float32') *
        tf.cast(
            tf.not_equal(
                tf.argmax(y_pred[:,:-1], axis=1, output_type='int32'),
                bot_index),
            'float32'))

def rtk_vra_cat(y_true, y_pred):
    labels = tf.argmax(y_true, axis=1, output_type='int32')[:,None]

    return rtk_vra_sparse(labels, y_pred)


def rtk_vra(y_true, y_pred):
    return tf.case([
        (tf.equal(tf.shape(y_true)[1], 1),
            lambda: rtk_vra_sparse(y_true, y_pred))],
        default=lambda: rtk_vra_cat(y_true, y_pred))


def affinity_vra_sparse(y_true, y_pred): return rtk_vra_sparse(y_true, y_pred)
def affinity_vra_cat(y_true, y_pred): return rtk_vra_cat(y_true, y_pred)
def affinity_vra(y_true, y_pred): return rtk_vra(y_true, y_pred)


def top_k_clean_acc(k, sparse=None):

    def metric_sparse(y_true, y_pred):
        return sparse_top_k_categorical_accuracy(y_true, y_pred[:, :-1], k)

    metric_sparse.__name__ = f'top_{k}_clean_acc_sparse'

    def metric_cat(y_true, y_pred):
        return top_k_categorical_accuracy(y_true, y_pred[:, :-1], k)

    metric_cat.__name__ = f'top_{k}_clean_acc_cat'

    def metric(y_true, y_pred):
        return tf.case([
            (tf.equal(tf.shape(y_true)[1], 1),
                lambda: metric_sparse(y_true, y_pred))],
            default=lambda: metric_cat(y_true, y_pred))

    metric.__name__ = f'top_{k}_clean_acc'

    if sparse is None:
        return metric

    else:
        return metric_sparse if sparse else metric_cat

def top_k_clean_acc_sparse(k):
    return top_k_clean_acc(k, sparse=True)

def top_k_clean_acc_cat(k):
    return top_k_clean_acc(k, sparse=False)