from keras.layers.core import Dense, Flatten
from keras.layers.convolutional import Conv2D, ZeroPadding2D


def layers(input_shape):
    return [
        ZeroPadding2D(padding=3, input_shape=input_shape, data_format='channels_first'),
        Conv2D(filters=48, kernel_size=(7, 7), data_format='channels_first', activation='relu'),

        ZeroPadding2D(padding=2, data_format='channels_first'),
        Conv2D(filters=32, kernel_size=(5, 5), data_format='channels_first', activation='relu'),

        ZeroPadding2D(padding=2, data_format='channels_first'),
        Conv2D(filters=32, kernel_size=(5, 5), data_format='channels_first', activation='relu'),

        ZeroPadding2D(padding=2, data_format='channels_first'),
        Conv2D(filters=32, kernel_size=(5, 5), data_format='channels_first', activation='relu'),

        Flatten(),
        Dense(512, activation='relu'),
    ]
