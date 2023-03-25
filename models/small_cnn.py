import os

from dlgo.preprocessing.processor import GoDataProcessor
from dlgo.encoders.oneplane import OnePlaneEncoder
from dlgo.networks import small
from keras.models import Sequential
from keras.layers.core import Dense
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from constants import ROOT_DIR

if __name__ == '__main__':

    print(tf.config.list_physical_devices('GPU'))
    go_board_rows, go_board_cols = 19, 19
    num_classes = go_board_rows * go_board_cols
    num_sample_games = 100
    data_dir = os.path.join(ROOT_DIR, "records/kgs/data")
    exp_dir = os.path.join(ROOT_DIR, "experiments/small_cnn_001")
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    encoder = OnePlaneEncoder((go_board_rows, go_board_cols))
    processor = GoDataProcessor(
        encoder=encoder.name(),
        data_dir=data_dir,
        exp_dir=exp_dir,
    )
    train_generator = processor.load_go_data('train', num_sample_games, use_generator=True)
    test_generator = processor.load_go_data('test', num_sample_games, use_generator=True)

    input_shape = (encoder.num_planes, go_board_rows, go_board_cols)
    network_layers = small.layers(input_shape)
    model = Sequential()
    for layer in network_layers:
        model.add(layer)
    model.add(Dense(num_classes, activation='relu'))
    model.compile(
        loss='categorical_crossentropy',
        optimizer='sgd',
        metrics=['accuracy'],
    )

    epochs = 5
    batch_size = 128
    model.fit(
        x=train_generator.generate(batch_size, num_classes),
        epochs=epochs,
        steps_per_epoch=train_generator.get_num_samples() / batch_size,
        validation_data=test_generator.generate(batch_size, num_classes),
        verbose=2,
        callbacks=[
            ModelCheckpoint(os.path.join(exp_dir, 'checkpoints/small_cnn_epoch_{epoch}.h5'))
        ]
    )

    model.evaluate_generator(
        generator=test_generator.generate(batch_size, num_classes),
        steps=test_generator.get_num_samples() / batch_size
    )
