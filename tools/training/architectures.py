from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.initializers import Orthogonal
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input

from gloro.layers import InvertibleDownsampling
from gloro.layers import MinMax
from gloro.layers import ResnetBlock


def _add_pool(z, pooling_type, activation=None, initialization='orthogonal'):
    if pooling_type == 'avg':
        return AveragePooling2D()(z)

    elif pooling_type == 'conv':
        channels = z.shape[-1]

        z = Conv2D(
            channels, 
            4, 
            strides=2, 
            padding='same', 
            kernel_initializer=initialization)(z)

        return _add_activation(z, activation)

    elif pooling_type == 'invertible':
        return InvertibleDownsampling()(z)

    else:
        raise ValueError(f'unknown pooling type: {pooling_type}')

def _add_activation(z, activation_type='relu'):
    if activation_type == 'relu':
        return Activation('relu')(z)

    elif activation_type == 'elu':
        return Activation('elu')(z)

    elif activation_type == 'softplus':
        return Activation('softplus')(z)

    elif activation_type == 'minmax':
        return MinMax()(z)

    else:
        raise ValueError(f'unknown activation type: {activation_type}')


def cnn_simple(
    input_shape, 
    num_classes, 
    pooling='conv', 
    activation='relu',
    initialization='orthogonal',
):
    x = Input(input_shape)
    z = Conv2D(32, 3, padding='same', kernel_initializer=initialization)(x)
    z = _add_activation(z, activation)
    z = _add_pool(z, pooling, activation, initialization)

    z = Conv2D(64, 3, padding='same', kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)
    z = _add_pool(z, pooling, activation, initialization)

    z = Flatten()(z)
    z = Dense(256, kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)

    z = Dense(256, kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)

    y = Dense(num_classes, kernel_initializer=initialization)(z)

    return x, y

def minmax_cnn_simple(
    input_shape, 
    num_classes, 
    pooling='invertible', 
    initialization='orthogonal',
    normalize_lc=False,
):
    return cnn_simple(
        input_shape, num_classes, 
        pooling=pooling, 
        activation='minmax',
        initialization=initialization)


def cnn_2C2F(
    input_shape, 
    num_classes, 
    activation='relu',
    initialization='orthogonal',
):
    x = Input(input_shape)
    z = Conv2D(
        16, 4, strides=2, padding='same', kernel_initializer=initialization)(x)
    z = _add_activation(z, activation)

    z = Conv2D(
        32, 4, strides=2, padding='same', kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)

    z = Flatten()(z)
    z = Dense(100, kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)
    y = Dense(num_classes, kernel_initializer=initialization)(z)

    return x, y

def minmax_cnn_2C2F(
    input_shape, 
    num_classes, 
    pooling='conv', 
    initialization='orthogonal',
):
    return cnn_2C2F(
        input_shape, num_classes, 
        pooling=pooling, 
        activation='minmax',
        initialization=initialization)


def cnn_4C3F(
    input_shape, 
    num_classes, 
    pooling='conv', 
    activation='relu',
    initialization='orthogonal',
):
    x = Input(input_shape)
    z = Conv2D(32, 3, padding='same', kernel_initializer=initialization)(x)
    z = _add_activation(z, activation)
    z = _add_pool(z, pooling, activation, initialization)

    z = Conv2D(64, 3, padding='same', kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)
    z = _add_pool(z, pooling, activation, initialization)

    z = Flatten()(z)
    z = Dense(512, kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)
    z = Dense(512, kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)
    y = Dense(num_classes, kernel_initializer=initialization)(z)

    return x, y

def minmax_cnn_4C3F(
    input_shape, 
    num_classes, 
    pooling='conv', 
    initialization='orthogonal',
):
    return cnn_4C3F(
        input_shape, num_classes, 
        pooling=pooling, 
        activation='minmax',
        initialization=initialization)


def cnn_6C2F(
    input_shape, 
    num_classes, 
    pooling='conv', 
    activation='relu',
    initialization='orthogonal',
):
    x = Input(input_shape)
    z = Conv2D(32, 3, padding='same', kernel_initializer=initialization)(x)
    z = _add_activation(z, activation)
    z = Conv2D(32, 3, padding='same', kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)
    z = _add_pool(z, pooling, activation, initialization)

    z = Conv2D(64, 3, padding='same', kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)
    z = Conv2D(64, 3, padding='same', kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)
    z = _add_pool(z, pooling, activation, initialization)

    z = Flatten()(z)
    z = Dense(512, kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)
    y = Dense(num_classes, kernel_initializer=initialization)(z)

    return x, y

def minmax_cnn_6C2F(
    input_shape, 
    num_classes, 
    pooling='conv', 
    initialization='orthogonal',
):
    return cnn_6C2F(
        input_shape, num_classes, 
        pooling=pooling, 
        activation='minmax',
        initialization=initialization)


def cnn_8C2F(
    input_shape, 
    num_classes, 
    pooling='conv', 
    activation='relu',
    initialization='orthogonal',
):
    x = Input(input_shape)
    z = Conv2D(64, 3, padding='same', kernel_initializer=initialization)(x)
    z = _add_activation(z, activation)
    z = Conv2D(64, 3, padding='same', kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)
    z = _add_pool(z, pooling, activation, initialization)

    z = Conv2D(128, 3, padding='same', kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)
    z = Conv2D(128, 3, padding='same', kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)
    z = _add_pool(z, pooling, activation, initialization)

    z = Conv2D(256, 3, padding='same', kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)
    z = _add_pool(z, pooling, activation, initialization)

    z = Flatten()(z)
    z = Dense(256, kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)
    y = Dense(num_classes, kernel_initializer=initialization)(z)

    return x, y

def minmax_cnn_8C2F(
    input_shape, 
    num_classes, 
    pooling='conv', 
    initialization='orthogonal',
):
    return cnn_8C2F(
        input_shape, num_classes, 
        pooling=pooling, 
        activation='minmax',
        initialization=initialization)


def alexnet(
    input_shape,
    num_classes,
    pooling='avg',
    activation='relu',
    initialization='orthogonal',
    dropout=False,
):
    x = Input(input_shape)

    z = Conv2D(
        96,
        11,
        padding='same',
        strides=4,
        kernel_initializer=initialization)(x)
    z = _add_activation(z, activation)
    z = _add_pool(z, pooling, activation, initialization)

    z = Conv2D(256, 5, padding='same', kernel_initializer=initialization)(z)
    z = Activation('relu')(z)
    z = _add_pool(z, pooling, activation, initialization)

    z = Conv2D(384, 3, padding='same', kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)
    z = Conv2D(384, 3, padding='same', kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)
    z = Conv2D(384, 3, padding='same', kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)
    z = _add_pool(z, pooling, activation, initialization)

    z = Flatten()(z)
    if dropout:
        z = Dropout(0.5)(z)
    z = Dense(4096, kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)
    if dropout:
        z = Dropout(0.5)(z)
    z = Dense(4096, kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)
    
    y = Dense(num_classes, kernel_initializer=initialization)(z)

    return x, y

def minmax_alexnet(
    input_shape,
    num_classes,
    pooling='invertible',
    initialization='orthogonal',
    dropout=False,
):
    return alexnet(
        input_shape, num_classes, 
        pooling=pooling,
        activation='minmax',
        initialization=initialization,
        dropout=dropout)


def vgg16(
    input_shape,
    num_classes,
    pooling='avg',
    activation='relu',
    initialization='orthogonal',
):
    x = Input(input_shape)

    z = Conv2D(64, 3, padding='same', kernel_initializer=initialization)(x)
    z = _add_activation(z, activation)
    z = Conv2D(64, 3, padding='same', kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)
    z = _add_pool(z, pooling, activation, initialization)

    z = Conv2D(128, 3, padding='same', kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)
    z = Conv2D(128, 3, padding='same', kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)
    z = _add_pool(z, pooling, activation, initialization)

    z = Conv2D(256, 3, padding='same', kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)
    z = Conv2D(256, 3, padding='same', kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)
    z = Conv2D(256, 3, padding='same', kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)
    z = _add_pool(z, pooling, activation, initialization)

    z = Conv2D(512, 3, padding='same', kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)
    z = Conv2D(512, 3, padding='same', kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)
    z = Conv2D(512, 3, padding='same', kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)
    z = _add_pool(z, pooling, activation, initialization)

    z = Conv2D(512, 3, padding='same', kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)
    z = Conv2D(512, 3, padding='same', kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)
    z = Conv2D(512, 3, padding='same', kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)
    z = _add_pool(z, pooling, activation, initialization)

    z = Flatten()(z)
    z = Dense(4096, kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)
    z = Dense(4096, kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)

    y = Dense(num_classes, kernel_initializer=initialization)(z)

    return x, y

def minmax_vgg16(
        input_shape,
        num_classes,
        pooling='invertible',
        initialization='orthogonal'):

    return vgg16(
        input_shape, num_classes, 
        pooling=pooling,
        activation='minmax',
        initialization=initialization)


def resnet_tiny(
    input_shape,
    num_classes,
    pooling='avg',
    activation='relu',
    initialization='orthogonal',
    fixup_residual_scaling=False,
    identity_skip=False,
):
    x = Input(input_shape)

    z = Conv2D(64, (5, 5), strides=(1, 1), padding='same')(x)
    z = _add_activation(z, activation)
    z = _add_pool(z, pooling, activation, initialization)

    z = ResnetBlock(
        filters=(128, 128, 128),
        kernel_sizes=(3, 3, 1),
        stride1=2,
        activation=activation,
        use_invertible_downsample=pooling == 'invertible',
        kernel_initializer=initialization,
        use_fixup_weight_and_bias=fixup_residual_scaling,
        identity_skip=identity_skip)(z)
    z = ResnetBlock(
        filters=(256, 256, 256),
        kernel_sizes=(3, 3, 1),
        stride1=2,
        activation=activation,
        use_invertible_downsample=pooling == 'invertible',
        kernel_initializer=initialization,
        use_fixup_weight_and_bias=fixup_residual_scaling,
        identity_skip=identity_skip)(z)
    z = ResnetBlock(
        filters=(512, 512, 512),
        kernel_sizes=(3, 3, 1),
        stride1=2,
        activation=activation,
        use_invertible_downsample=pooling == 'invertible',
        kernel_initializer=initialization,
        use_fixup_weight_and_bias=fixup_residual_scaling,
        identity_skip=identity_skip)(z)

    z = _add_pool(z, pooling, activation, initialization)

    z = Flatten()(z)
    z = Dense(512, kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)

    y = Dense(num_classes, kernel_initializer=initialization)(z)

    return x, y

def minmax_resnet_tiny(
    input_shape,
    num_classes,
    pooling='invertible',
    initialization='orthogonal',
    fixup_residual_scaling=False,
    identity_skip=False,
):
    return resnet_tiny(
        input_shape, num_classes, 
        pooling=pooling, 
        activation='minmax',
        initialization=initialization,
        fixup_residual_scaling=fixup_residual_scaling,
        identity_skip=identity_skip)
