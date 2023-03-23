from keras.layers.core import Dense, Flatten
from keras.layers.convolutional import Conv2D, ZeroPadding2D


def layers(input_shape):
    return [
        ZeroPadding2D((2, 2), input_shape=input_shape, data_format='channels_first'),
        Conv2D(filters=64, kernel_size=(5, 5), padding='valid', data_format='channels_first', activation='relu'),

        ZeroPadding2D((2, 2), data_format='channels_first'),
        Conv2D(filters=64, kernel_size=(5, 5), data_format='channels_first', activation='relu'),

        ZeroPadding2D((1, 1), data_format='channels_first'),
        Conv2D(filters=64, kernel_size=(3, 3), data_format='channels_first', activation='relu'),

        ZeroPadding2D((1, 1), data_format='channels_first'),
        Conv2D(filters=64, kernel_size=(3, 3), data_format='channels_first', activation='relu'),

        ZeroPadding2D((1, 1), data_format='channels_first'),
        Conv2D(filters=64, kernel_size=(3, 3), data_format='channels_first', activation='relu'),

        Flatten(),
        Dense(512, activation='relu'),
    ]
