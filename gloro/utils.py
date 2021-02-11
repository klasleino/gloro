import tensorflow as tf
import tensorflow.keras.backend as K

from scipy.stats import truncnorm


def l2_normalize(x):
    return x / (tf.sqrt(tf.reduce_sum(x**2.)) + K.epsilon())


def np_truncated_normal(shape):
    return truncnorm.rvs(-2., 2., size=shape).astype('float32')


def global_lipschitz_bound_spectral_power(
        model, 
        num_iterations, 
        norm='l2', 
        exclude_last_layer=True, 
        maintain_state=True):

    k = 1.

    while_cond = lambda i, _: i < num_iterations

    layers = model.layers[:-1] if exclude_last_layer else model.layers

    # Used if we're maintaining the iterates' states.
    iterates = []
    updates = []

    for layer in layers:
        if not hasattr(layer, 'kernel'):
            continue

        W = layer.kernel

        if len(W.shape) == 4:
            # This is a convolutional layer.
            if norm == 'l2':
                if maintain_state:
                    x_0 = tf.Variable(
                        np_truncated_normal((1,*layer.input_shape[1:])),
                        trainable=False)

                    iterates.append(x_0)

                else:
                    x_0 = tf.random.truncated_normal(
                        shape=(1,*layer.input_shape[1:]))

                def body(i, x):
                    x = l2_normalize(x)

                    x_p = tf.nn.conv2d(
                        x,
                        W,
                        strides=layer.strides,
                        padding=layer.padding.upper())
                    x = tf.nn.conv2d_transpose(
                        x_p,
                        W,
                        x.shape,
                        strides=layer.strides,
                        padding=layer.padding.upper())

                    return i + 1, x

                _, x = tf.while_loop(while_cond, body, [tf.constant(0), x_0])

                if maintain_state:
                    updates.append((x_0, x))

                Wx = tf.nn.conv2d(
                    x, W, strides=layer.strides, padding=layer.padding.upper())

                k *= tf.sqrt(tf.reduce_sum(Wx**2.) / tf.reduce_sum(x**2.))

            else:
                raise ValueError(f'{norm} norm not supported')

        else:
            if norm == 'l2':
                if maintain_state:
                    x_0 = tf.Variable(
                        np_truncated_normal((W.shape[1], 1)),
                        trainable=False)

                    iterates.append(x_0)

                else:
                    x_0 = tf.random.truncated_normal(shape=(W.shape[1], 1))

                def body(i, x):
                    x = l2_normalize(x)
                    x_p = W @ x
                    x = tf.transpose(W) @ x_p

                    return i + 1, x

                _, x = tf.while_loop(while_cond, body, [tf.constant(0), x_0])

                if maintain_state:
                    updates.append((x_0, x))

                k *= tf.sqrt(tf.reduce_sum((W @ x)**2.) / tf.reduce_sum(x**2.))

            else:
                raise ValueError(f'{norm} norm not supported')

    if maintain_state:
        return k, iterates, updates

    return k
