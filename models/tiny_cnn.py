import os

from dlgo.preprocessing.processor import GoDataProcessor
from dlgo.encoders.oneplane import OnePlaneEncoder
#from dlgo.networks import small
from keras.models import Sequential
from keras.layers.core import Dense
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from constants import ROOT_DIR
from keras.models import Sequential
from keras.layers import (
    Conv2D,
    Dense,
    Flatten,
    MaxPooling2D,
    Dropout,
)

if __name__ == '__main__':
    # Check GPU
    print(tf.config.list_physical_devices('GPU'))

    # Configs
    go_board_rows, go_board_cols = 19, 19
    num_classes = go_board_rows * go_board_cols
    num_sample_games = 100
    data_dir = os.path.join(ROOT_DIR, "records/kgs/data")
    exp_dir = os.path.join(ROOT_DIR, "experiments/tiny_cnn_001")
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
        os.makedirs(os.path.join(exp_dir, "checkpoints"))
        os.makedirs(os.path.join(exp_dir, "train"))

    # Preprocessing pipeline
    encoder = OnePlaneEncoder((go_board_rows, go_board_cols))
    processor = GoDataProcessor(
        encoder=encoder.name(),
        data_dir=data_dir,
        exp_dir=exp_dir,
    )
    train_generator = processor.load_go_data('train', num_sample_games, use_generator=True)
    test_generator = processor.load_go_data('test', num_sample_games, use_generator=True)

    # Model
    input_shape = (encoder.num_planes, go_board_rows, go_board_cols)
    model = Sequential()
    model.add(
        Conv2D(
            filters=48,
            kernel_size=(3, 3),
            activation='relu',
            padding='same',
            data_format='channels_first',
            input_shape=input_shape,
        )
    )
    model.add(Dropout(rate=0.5))
    model.add(
        Conv2D(
            filters=48,
            kernel_size=(3, 3),
            padding='same',
            data_format='channels_first',
            activation='relu',
        )
    )
    model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_first'))
    model.add(Dropout(rate=0.5))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(
        loss='categorical_crossentropy',
        optimizer='sgd',
        metrics=['accuracy'],
    )

    # Training
    epochs = 5
    batch_size = 128
    print(f"Total Number of training data: {train_generator.get_num_samples()}")
    print(f"Total Number of steps per epoch: {train_generator.get_num_samples() / batch_size}")
    model.fit(
        x=train_generator.generate(batch_size, num_classes),
        epochs=epochs,
        steps_per_epoch=int(train_generator.get_num_samples() // batch_size),
        validation_data=test_generator.generate(batch_size, num_classes),
        validation_steps=int(test_generator.get_num_samples() // batch_size),
        verbose=1,
        callbacks=[
            ModelCheckpoint(os.path.join(exp_dir, 'checkpoints/tiny_cnn_epoch_{epoch}.h5'))
        ]
    )

    model.evaluate(
        x=test_generator.generate(batch_size, num_classes),
        steps=int(test_generator.get_num_samples() // batch_size),
    )

    model.summary(show_trainable=True)
