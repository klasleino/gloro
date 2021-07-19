import numpy as np
import tensorflow as tf

from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.utils import Progbar

from gloro.constants import EPS


def lipschitz_lb(f, X1, X2, iterations=1000, verbose=True):

    optimizer = Adam(lr=0.0001)

    X1 = tf.Variable(X1, name='x1', dtype='float32')
    X2 = tf.Variable(X2, name='x2', dtype='float32')
    
    max_L = None

    if verbose:
        pb = Progbar(iterations, stateful_metrics=['LC'])
    
    for _ in range(iterations):
        with tf.GradientTape() as tape:
            y1 = f(X1)
            y2 = f(X2)
            
            # The definition of the margin is not entirely symmetric: the top
            # class must remain the same when measuring both points. We assume
            # X1 is the reference point for determining the top class.
            original_predictions = tf.cast(
                tf.equal(y1, tf.reduce_max(y1, axis=1, keepdims=True)), 
                'float32')
            
            # This takes the logit at the top class for both X1 and X2.
            y1_j = tf.reduce_sum(
                y1 * original_predictions, axis=1, keepdims=True)
            y2_j = tf.reduce_sum(
                y2 * original_predictions, axis=1, keepdims=True)
            
            margin1 = y1_j - y1
            margin2 = y2_j - y2

            axes = tuple((tf.range(len(X1.shape) - 1) + 1).numpy())
            
            L = tf.abs(margin1 - margin2) / (tf.sqrt(
                tf.reduce_sum((X1 - X2)**2, axis=axes)) + EPS)[:,None]

            loss = -tf.reduce_max(L, axis=1)
            
        grad = tape.gradient(loss, [X1, X2])

        optimizer.apply_gradients(zip(grad, [X1, X2]))
        
        if max_L is None:
            max_L = L
        else:
            max_L = tf.maximum(max_L, L)

        if verbose:
            pb.add(1, [('LC', tf.reduce_max(max_L))])
        
    return tf.reduce_max(max_L)


def local_lipschitz_lb(f, X1, X2, eps, iterations=1000, verbose=True):

    optimizer = Adam(lr=0.0001)

    X0 = tf.constant(X1, dtype='float32')
    y0 = f(X0)

    X1 = tf.Variable(X1, name='x1', dtype='float32')
    X2 = tf.Variable(X2, name='x2', dtype='float32')
    
    max_L = None

    if verbose:
        pb = Progbar(iterations, stateful_metrics=['max_LC', 'mean_LC'])
    
    for i in range(iterations):
        with tf.GradientTape() as tape:

            axes = tuple((tf.range(len(X1.shape) - 1) + 1).numpy())

            # Project so that X1 and X2 are at distance at most `eps` from X0.
            delta1 = X1 - X0
            dist1 = tf.sqrt(tf.reduce_sum(
                delta1 * delta1, axis=axes, keepdims=True))
            
            delta2 = X2 - X0
            dist2 = tf.sqrt(tf.reduce_sum(
                delta2 * delta2, axis=axes, keepdims=True))
            
            # Only project if `dist` > `eps`.
            where_dist_gt_eps1 = tf.cast(dist1 > eps, 'float32')
            where_dist_gt_eps2 = tf.cast(dist2 > eps, 'float32')

            X1.assign(
                (X0 + eps * delta1 / (dist1 + EPS)) * where_dist_gt_eps1 + 
                X1 * (1 - where_dist_gt_eps1))
            X2.assign(
                (X0 + eps * delta2 / (dist2 + EPS)) * where_dist_gt_eps2 + 
                X2 * (1 - where_dist_gt_eps2))

            y1 = f(X1)
            y2 = f(X2)
            
            # The definition of the margin is not entirely symmetric: the top
            # class must remain the same when measuring both points. We assume
            # X0 is the reference point for determining the top class.
            original_predictions = tf.cast(
                tf.equal(y0, tf.reduce_max(y0, axis=1, keepdims=True)), 
                'float32')
            
            # This takes the logit at the top class for both X1 and X2.
            y1_j = tf.reduce_sum(
                y1 * original_predictions, axis=1, keepdims=True)
            y2_j = tf.reduce_sum(
                y2 * original_predictions, axis=1, keepdims=True)
            
            margin1 = y1_j - y1
            margin2 = y2_j - y2

            L = tf.abs(margin1 - margin2) / (tf.sqrt(
                tf.reduce_sum((X1 - X2)**2, axis=axes)) + EPS)[:,None]

            loss = -tf.reduce_max(L, axis=1)
           
        if i < iterations - 1: 
            grad = tape.gradient(loss, [X1, X2])

            optimizer.apply_gradients(zip(grad, [X1, X2]))
        
        if max_L is None:
            max_L = L
        else:
            max_L = tf.maximum(max_L, L)

        if verbose:
            metrics = [
                ('max_LC', tf.reduce_max(max_L)), 
                ('mean_LC', tf.reduce_mean(max_L))
            ]
            
            pb.add(1, metrics)
        
    return tf.reduce_max(max_L), tf.reduce_mean(max_L)
