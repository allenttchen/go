import glob
import numpy as np
from keras.utils import to_categorical


class DataGenerator:
    def __init__(self, train_data_dir, samples):
        self.train_data_dir = train_data_dir
        self.samples = samples
        self.files = set(file_name for file_name, index in samples)
        self.num_samples = None # number of moves

    def get_num_samples(self, batch_size=128, num_classes=19*19):
        """
        Get number of moves you have sampled and preprocessed earlier
        """
        if self.num_samples is not None:
            return self.num_samples
        else:
            self.num_samples = 0
            for X, y in self._generate(batch_size=batch_size, num_classes=num_classes):
                self.num_samples += X.shape[0]
            return self.num_samples

    def _generate(self, batch_size, num_classes):
        """
        Load features and labels from the sampled and preprocessed .npy features and labels files
        """
        # iterate through each file from samples
        for zip_file_name in self.files:
            file_name = zip_file_name.replace('.tar.gz', '') + 'train'
            base = self.train_data_dir + '/' + file_name + '_features_*.npy'
            # iterate through each game
            for feature_file in glob.glob(base):
                label_file = feature_file.replace('features', 'labels')
                # Load all features and labels
                x = np.load(feature_file)
                y = np.load(label_file)
                x = x.astype('float32')
                y = to_categorical(y.astype(int), num_classes)
                # Generate batches
                # Note: this truncates any remaining moves <batch_size in the .npy file
                while x.shape[0] >= batch_size:
                    x_batch, x = x[:batch_size], x[batch_size:]
                    y_batch, y = y[:batch_size], y[batch_size:]
                    yield x_batch, y_batch

    def generate(self, batch_size=128, num_classes=19*19):
        while True:
            for item in self._generate(batch_size, num_classes):
                yield item
