from gloro.layers import AveragePooling2D
from gloro.layers import Conv2D
from gloro.layers import Dense
from gloro.layers import Flatten
from gloro.layers import Input
from gloro.layers import InvertibleDownsampling
from gloro.layers import LiResNetBlock
from gloro.layers import MaxPooling2D
from gloro.layers import MinMax


def Activation(activation_type='minmax'):
    if activation_type == 'minmax':
        return MinMax()

    if activation_type == 'relu':
        return ReLU()

    else:
        raise ValueError(f'unknown activation type: {activation_type}')

def Downsample(downsample_type):
    if downsample_type == 'invertible':
        return InvertibleDownsampling(2)

    elif downsample_type == 'avg':
        return AveragePooling2D(2)

    elif downsample_type == 'max':
        return MaxPooling2D(2)

    else:
        raise ValueError(f'unknown downsample type: {downsample_type}')


def cnn_2C2F(
    input_shape, 
    num_classes, 
    activation='minmax',
    initialization='orthogonal',
):
    x = Input(input_shape)
    z = Conv2D(
        16, 4,
        strides=2,
        padding='same',
        kernel_initializer=initialization,
    )(x)
    z = Activation(activation)(z)

    z = Conv2D(
        32, 4,
        strides=2,
        padding='same',
        kernel_initializer=initialization,
    )(z)
    z = Activation(activation)(z)

    z = Flatten()(z)
    z = Dense(100, kernel_initializer=initialization)(z)
    z = Activation(activation)(z)

    y = Dense(num_classes, kernel_initializer=initialization)(z)

    return x, y


def cnn_4C3F(
    input_shape, 
    num_classes, 
    pooling='conv', 
    activation='minmax',
    initialization='orthogonal',
):
    x = Input(input_shape)

    z = Conv2D(32, 3, padding='same', kernel_initializer=initialization)(x)
    z = Activation(activation)(z)
    z = Conv2D(
        32, 4,
        strides=2,
        padding='same',
        kernel_initializer=initialization,
    )(z)
    z = Activation(activation)(z)

    z = Conv2D(64, 3, padding='same', kernel_initializer=initialization)(z)
    z = Activation(activation)(z)
    z = Conv2D(
        64, 4,
        strides=2,
        padding='same',
        kernel_initializer=initialization,
    )(z)
    z = Activation(activation)(z)

    z = Flatten()(z)
    z = Dense(512, kernel_initializer=initialization)(z)
    z = Activation(activation)(z)
    z = Dense(512, kernel_initializer=initialization)(z)
    z = z = Activation(activation)(z)

    y = Dense(num_classes, kernel_initializer=initialization)(z)

    return x, y


def cnn_6C2F(
    input_shape, 
    num_classes, 
    activation='minmax',
    initialization='orthogonal',
):
    x = Input(input_shape)

    z = Conv2D(32, 3, padding='same', kernel_initializer=initialization)(x)
    z = Activation(activation)(z)
    z = Conv2D(32, 3, padding='same', kernel_initializer=initialization)(z)
    z = Activation(activation)(z)
    z = Conv2D(
        32, 4,
        strides=2,
        padding='same',
        kernel_initializer=initialization,
    )(z)
    z = Activation(activation)(z)

    z = Conv2D(64, 3, padding='same', kernel_initializer=initialization)(z)
    z = Activation(activation)(z)
    z = Conv2D(64, 3, padding='same', kernel_initializer=initialization)(z)
    z = Activation(activation)(z)
    z = Conv2D(
        64, 4,
        strides=2,
        padding='same',
        kernel_initializer=initialization,
    )(z)
    z = Activation(activation)(z)

    z = Flatten()(z)
    z = Dense(512, kernel_initializer=initialization)(z)
    z = Activation(activation)(z)

    y = Dense(num_classes, kernel_initializer=initialization)(z)

    return x, y


def cnn_8C2F(
    input_shape, 
    num_classes, 
    activation='minmax',
    initialization='orthogonal',
):
    x = Input(input_shape)

    z = Conv2D(64, 3, padding='same', kernel_initializer=initialization)(x)
    z = Activation(activation)(z)
    z = Conv2D(64, 3, padding='same', kernel_initializer=initialization)(z)
    z = Activation(activation)(z)
    z = Conv2D(
        64, 4,
        strides=2,
        padding='same',
        kernel_initializer=initialization,
    )(z)
    z = Activation(activation)(z)

    z = Conv2D(128, 3, padding='same', kernel_initializer=initialization)(z)
    z = Activation(activation)(z)
    z = Conv2D(128, 3, padding='same', kernel_initializer=initialization)(z)
    z = Activation(activation)(z)
    z = Conv2D(
        128, 4,
        strides=2,
        padding='same',
        kernel_initializer=initialization,
    )(z)
    z = Activation(activation)(z)

    z = Conv2D(256, 3, padding='same', kernel_initializer=initialization)(z)
    z = Activation(activation)(z)
    z = Conv2D(
        256, 4,
        strides=2,
        padding='same',
        kernel_initializer=initialization,
    )(z)
    z = Activation(activation)(z)

    z = Flatten()(z)
    z = Dense(256, kernel_initializer=initialization)(z)
    z = Activation(activation)(z)

    y = Dense(num_classes, kernel_initializer=initialization)(z)

    return x, y


def _cnn_CxCC2F(
    backbone_depth,
    backbone_width,
    input_shape,
    num_classes,
    activation='minmax',
    initialization='orthogonal',
    stem_downsample=2,
):
    x = Input(input_shape)

    # Stem.
    z = Conv2D(
        backbone_width, 5,
        strides=stem_downsample,
        padding='same',
        kernel_initializer=initialization,
    )(x)
    z = Activation(activation)(z)

    # Backbone.
    for _ in range(backbone_depth):
        z = Conv2D(
            backbone_width, 3,
            padding='same',
            kernel_initializer=initialization,
        )(z)
        z = Activation(activation)(z)

    # Neck.
    z = Conv2D(
        2 * backbone_width, 4,
        strides=4,
        padding='valid',
        kernel_initializer=initialization,
    )(z)
    z = Activation(activation)(z)

    z = Flatten()(z)
    z = Dense(512)(z)
    z = Activation(activation)(z)

    # Head.
    y = Dense(num_classes)(z)

    return x, y

def cnn_C6CC2F(
    input_shape,
    num_classes,
    width=128,
    activation='minmax',
    initialization='orthogonal',
    stem_downsample=2,
):
    return _cnn_CxCC2F(
        6, width, input_shape, num_classes,
        activation=activation,
        initialization=initialization,
        stem_downsample=stem_downsample,
    )


def _liresnet_CxCC2F(
    backbone_depth,
    backbone_width,
    input_shape,
    num_classes,
    activation='minmax',
    initialization='orthogonal',
    stem_downsample=2,
):
    x = Input(input_shape)

    # Stem.
    z = Conv2D(
        backbone_width, 5,
        strides=stem_downsample,
        padding='same',
        kernel_initializer=initialization,
    )(x)
    z = Activation(activation)(z)

    # Backbone.
    for _ in range(backbone_depth):
        z = LiResNetBlock(
            3,
            residual_scale=backbone_depth**(-0.5),
            kernel_initializer=initialization,
        )(z)
        z = Activation(activation)(z)

    # Neck.
    z = Conv2D(
        2 * backbone_width, 4,
        strides=4,
        padding='valid',
        kernel_initializer=initialization,
    )(z)
    z = Activation(activation)(z)

    z = Flatten()(z)
    z = Dense(512)(z)
    z = Activation(activation)(z)

    # Head.
    y = Dense(num_classes)(z)

    return x, y

def liresnet_C6CC2F(
    input_shape,
    num_classes,
    width=128,
    activation='minmax',
    initialization='orthogonal',
    stem_downsample=2,
):
    return _liresnet_CxCC2F(
        6, width, input_shape, num_classes,
        activation=activation,
        initialization=initialization,
        stem_downsample=stem_downsample,
    )

def liresnet_C18CC2F(
    input_shape,
    num_classes,
    width=256,
    activation='minmax',
    initialization='orthogonal',
    stem_downsample=2,
):
    return _liresnet_CxCC2F(
        18, width, input_shape, num_classes,
        activation=activation,
        initialization=initialization,
        stem_downsample=stem_downsample,
    )

