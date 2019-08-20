from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Input
from keras.models import Model

import numpy as np

from .util import inst_layers


def Conv(filters, kernel_size=(3, 3), activation='relu', input_shape=None):
    """
    # Convolution 2D layer
    """
    if input_shape:
        return Conv2D(filters=filters,
                      kernel_size=kernel_size,
                      padding='same',
                      activation=activation,
                      input_shape=input_shape)
    else:
        return Conv2D(filters=filters,
                      kernel_size=kernel_size,
                      padding='same',
                      activation=activation)


def Model2D(input_shape, num_classes, include_top=True):
    """
    # Define Model
    # Arguments
        input_shape: (height, width, channel)
        num_classes: number of classes
    """

    small_size = min(input_shape[:2])
    nb_convolution = 0

    while small_size > 8:
        small_size = small_size // 2
        nb_convolution += 1

    layers = [
        Conv(8, input_shape=input_shape),
        MaxPooling2D()
    ]

    layers += [
        [
            Conv(8 * 2**layer_id),
            BatchNormalization(),
            MaxPooling2D(),
        ]
        for layer_id in range(1, nb_convolution)
    ]

    latent_dim = np.prod(input_shape[:2])  # width * height
    latent_dim *= 8 * 2 ** (nb_convolution - 1)  # filter size at last convolution
    latent_dim //= 4 ** nb_convolution  # number of training parameters reduced with convolution
    latent_dim //= 4  # dimension reduction from flatten to dense

    if include_top:

        layers += [
            Flatten(),
            Dropout(0.25),
            Dense(latent_dim, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax', name='prediction')
        ]

    x_in = Input(shape=input_shape, name='input')
    prediction = inst_layers(layers, x_in)
    model = Model(x_in, prediction)

    return model
