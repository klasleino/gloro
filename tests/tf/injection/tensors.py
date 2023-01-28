import numpy as np
import tensorflow as tf


class Tensors(object):
    def to_np(self, x):
        return x.numpy()

    def norm(self, x):
        return tf.sqrt(tf.reduce_sum(x**2))

    def random(self, shape):
        return tf.random.normal(shape)

    def margin_jacobian(self, f, X):

        with tf.GradientTape() as tape:
            tape.watch(X)

            y = f(X)

            # This is the logit of the top (predicted) class at X.
            y_j = tf.reduce_max(y, axis=1, keepdims=True)

            margin = y_j - y

        jacobian = tape.jacobian(margin, X)

        N, num_classes = jacobian.shape[:2]

        # Reorder the dimensions and flatten the input shape.
        jacobian = tf.transpose(
            tf.reshape(jacobian, (N, num_classes, N, -1)),
            (0, 2, 1, 3),
        )

        # The Jacobian is only meaningful in index (i, i) of the N x N first two
        # dimensions, since the diagonal of entries here are the only ones that
        # correspond to outputs and inputs that are connected.
        return tf.gather_nd(jacobian, np.tile(np.arange(N)[:,None], 2))
