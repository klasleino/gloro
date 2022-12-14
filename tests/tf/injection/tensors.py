import tensorflow as tf


class Tensors(object):
    def to_np(self, x):
        return x.numpy()

    def norm(self, x):
        return tf.sqrt(tf.reduce_sum(x**2))

    def random(self, shape):
        return tf.random.normal(shape)
