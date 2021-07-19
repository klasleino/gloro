import numpy as np
import os
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.data import Dataset
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD


def data_augmentation(
        flip=True, 
        saturation=(0.5, 1.2), 
        contrast=(0.8, 1.2), 
        zoom=0.1,
        noise=None):
    
    def augment(x, y):
        batch_size = tf.shape(x)[0]
        input_shape = x.shape[1:]

        # Horizontal flips
        if flip:
            x = tf.image.random_flip_left_right(x)

        # Randomly adjust the saturation and contrast.
        if saturation is not None and input_shape[-1] == 3:
            x = tf.image.random_saturation(
                x, lower=saturation[0], upper=saturation[1])

        if contrast is not None:
            x = tf.image.random_contrast(
                x, lower=contrast[0], upper=contrast[1])

        # Randomly zoom.
        if zoom is not None:
            widths = tf.random.uniform([batch_size], 1. - zoom, 1.)
            top_corners = tf.random.uniform(
                [batch_size, 2], 0, 1. - widths[:, None])
            bottom_corners = top_corners + widths[:, None]
            boxes = tf.concat((top_corners, bottom_corners), axis=1)

            x = tf.image.crop_and_resize(
                x, boxes,
                box_indices=tf.range(batch_size),
                crop_size=input_shape[0:2])

        if noise is not None:
            x = x + tf.random.normal(tf.shape(x), stddev=noise)

        return x, y
    
    return augment


def get_data(dataset, batch_size, augmentation=None):

    # Get the augmentation.
    if augmentation is None or augmentation.lower() == 'none':
        augmentation = data_augmentation(
            flip=False, saturation=None, contrast=None, zoom=None)

    elif augmentation == 'all':
        augmentation = data_augmentation()

    elif augmentation == 'standard':
        augmentation = data_augmentation(
            saturation=None, contrast=None, zoom=0.25)

    elif augmentation == 'no_flip':
        augmentation = data_augmentation(
            flip=False, saturation=None, contrast=None, zoom=0.25)

    elif augmentation.startswith('rs'):
        flip = data_augmentation.startswith('rs_no_flip')

        if augmentation.startswith('rs.'):
            noise = float(augmentation.split('rs.')[1])

        elif augmentation.startswith('rs_no_flip.'):
            noise = float(augmentation.split('rs_no_flip.')[1])

        else:
            noise = 0.125

        augmentation = data_augmentation(
            flip=flip, saturation=None, contrast=None, noise=noise)

    else:
        raise ValueError(f'unknown augmentation type: {augmentation}')

    # Load the data.
    if dataset == 'tiny-imagenet':

        if 'TINY_IMAGENET_LOCATION' in os.environ:
            tiny_imagenet_dir = os.environ['TINY_IMAGENET_LOCATION']

        else:
            raise ValueError(
                'to use tiny-imagenet you must set the '
                '"TINY_IMAGENET_LOCATION" environment variable')

        x_tr = np.load(f'{tiny_imagenet_dir}/TinyImagenet_train_x.npy')
        x_te = np.load(f'{tiny_imagenet_dir}/TinyImagenet_test_x.npy')
        y_tr = np.load(f'{tiny_imagenet_dir}/TinyImagenet_train_y.npy')
        y_te = np.load(f'{tiny_imagenet_dir}/TinyImagenet_test_y.npy')

        train = (Dataset.from_tensor_slices((x_tr, y_tr))
            .cache()
            .batch(batch_size)
            .map(
                augmentation, 
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
                deterministic=False)
            .prefetch(tf.data.experimental.AUTOTUNE))

        test = (Dataset.from_tensor_slices((x_te, y_te))
            .cache()
            .batch(batch_size)
            .prefetch(tf.data.experimental.AUTOTUNE))

        metadata = lambda: None
        metadata.features = {
            'image': lambda: None,
            'label': lambda: None,
        }
        metadata.features['label'].num_classes = 200
        metadata.features['image'].shape = (64, 64, 3)

    else: 
        tfds_dir = os.environ['TFDS_DIR'] if 'TFDS_DIR' in os.environ else None

        split = ['train', 'test']

        (train, test), metadata = tfds.load(
            dataset,
            data_dir=tfds_dir,
            split=split, 
            with_info=True, 
            shuffle_files=True, 
            as_supervised=True)

        train = (train
            .map(
                lambda x,y: (tf.cast(x, 'float32') / 255., y), 
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
                deterministic=False)
            .cache()
            .batch(batch_size)
            .map(
                augmentation, 
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
                deterministic=False)
            .prefetch(tf.data.experimental.AUTOTUNE))

        test = (test
            .map(
                lambda x,y: (tf.cast(x, 'float32') / 255., y), 
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
                deterministic=False)
            .cache()
            .batch(batch_size)
            .prefetch(tf.data.experimental.AUTOTUNE))

    return train, test, metadata


def get_optimizer(optimizer, lr):
    if optimizer == 'adam':
        return Adam(lr=lr)

    elif optimizer.startswith('sgd'):
        if optimizer.startswith('sgd.'):
            return SGD(lr=lr, momentum=float(optimizer.split('sgd.')[1]))

        else:
            return SGD(lr=lr)

    else:
        raise ValueError(f'unknown optimizer: {optimizer}')
