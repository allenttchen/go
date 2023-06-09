import os
import tarfile
import gzip
import glob
import shutil
import multiprocessing
import sys

import numpy as np
from keras.utils import to_categorical
from dlgo.gosgf import Sgf_game
from dlgo.goboard import Board, GameState, Move
from dlgo.gotypes import Player, Point
from dlgo.encoders.base import get_encoder_by_name
from dlgo.preprocessing.index_processor import KGSIndex
from dlgo.preprocessing.sampling import Sampler
from dlgo.preprocessing.generator import DataGenerator
from constants import KGS_INDEX


def worker(jobinfo):
    try:
        clazz, raw_data_dir, exp_dir, train_data_dir, encoder, zip_file, data_file_name, game_list = jobinfo
        (
            clazz(raw_data_dir=raw_data_dir, exp_dir=exp_dir, train_data_dir=train_data_dir, encoder=encoder)
            .process_zip(zip_file, data_file_name, game_list)
        )

    except (KeyboardInterrupt, SystemExit):
        raise Exception('>>> Exiting child process.')


class GoDataProcessor:
    """
    Preprocessing pipeline from raw data to trains and labels numpy array or generators
    """
    def __init__(
        self,
        raw_data_dir,
        exp_dir,
        train_data_dir,
        index_page=KGS_INDEX,
        encoder='oneplane',
        seed=1337,
    ):
        self.encoder_string = encoder
        self.encoder = get_encoder_by_name(encoder, 19)
        self.index_page = index_page
        self.seed = seed
        self.raw_data_dir = raw_data_dir
        self.exp_dir = exp_dir
        self.train_data_dir = train_data_dir

    def load_go_data(
        self,
        data_type='train',
        num_sample_games=1000,
        use_generator=False,
    ):
        """
        In a parallel fashion, create features and labels .npy files from downloaded .tar.gz files
        Main method to get X and y
        """
        # download data from KGS
        index = KGSIndex(data_directory=self.raw_data_dir, index_page=self.index_page)
        index.download_files()

        # sample data
        sampler = Sampler(raw_data_dir=self.raw_data_dir, exp_dir=self.exp_dir, seed=self.seed)
        data = sampler.draw_data(data_type, num_sample_games)

        # Parallelize game extraction from zip files
        if len(os.listdir(self.train_data_dir)) == 0:
            self.map_to_workers(data_type, data)

        # use generator or load everything at once
        if use_generator:
            generator = DataGenerator(self.train_data_dir, data)
            return generator
        else:
            features_and_labels = self.consolidate_games(data_type, data)
            return features_and_labels

    def load_go_data_sequentially(self, data_type='train', num_sample_games=1000):
        """
        In an non-parallel fashion, create features and labels .npy files from downloaded .tar.gz files
        """
        # download data from KGS
        index = KGSIndex(data_directory=self.raw_data_dir, index_page=self.index_page)
        index.download_files()

        # sample data
        sampler = Sampler(raw_data_dir=self.raw_data_dir, exp_dir=self.exp_dir)
        data = sampler.draw_data(data_type, num_sample_games)

        # create mapping filename: the list of indices
        zip_names = set()
        indices_by_zip_names = {}
        for filename, index in data:
            zip_names.add(filename)
            if filename not in indices_by_zip_names:
                indices_by_zip_names[filename] = []
            indices_by_zip_names[filename].append(index)

        # extract the necessary game records from the zipped file
        for zip_name in zip_names:
            base_name = zip_name.replace('.tar.gz', '')
            data_file_name = base_name + data_type
            if not os.path.isfile(self.raw_data_dir + '/' + data_file_name):
                self.process_zip(zip_name, data_file_name, indices_by_zip_names[zip_name])

        features_and_labels = self.consolidate_games(data_type, data)
        return features_and_labels

    def unzip_data(self, zip_file_name):
        this_gz = gzip.open(self.raw_data_dir + '/' + zip_file_name)

        tar_file = zip_file_name[0:-3]
        this_tar = open(self.raw_data_dir + '/' + tar_file, 'wb')

        shutil.copyfileobj(this_gz, this_tar)
        this_tar.close()
        return tar_file

    def process_zip(self, zip_file_name, data_file_name, game_list):
        tar_file = self.unzip_data(zip_file_name)
        zip_file = tarfile.open(self.raw_data_dir + '/' + tar_file)
        name_list = zip_file.getnames()
        # Get total number of moves in this zip file
        total_examples = self.num_total_examples(zip_file, game_list, name_list)

        # initialize features and labels
        shape = self.encoder.shape()
        feature_shape = np.insert(shape, 0, np.asarray([total_examples]))
        features = np.zeros(feature_shape)
        labels = np.zeros((total_examples,))

        # Loop through each game in game_list
        counter = 0
        for index in game_list:
            name = name_list[index + 1]
            if not name.endswith('.sgf'):
                raise ValueError(name + ' is not a valid sgf')
            sgf_content = zip_file.extractfile(name).read()
            sgf = Sgf_game.from_string(sgf_content)

            game_state, first_move_done = self.get_handicap(sgf)

            # Iterates over all moves
            for item in sgf.main_sequence_iter():
                color, move_tuple = item.get_move()
                point = None
                if color is not None:
                    if move_tuple is not None:
                        row, col = move_tuple
                        point = Point(row + 1, col + 1)
                        move = Move.play(point)
                    else:
                        move = Move.pass_turn()
                    if first_move_done and point is not None:
                        features[counter] = self.encoder.encode(game_state)
                        labels[counter] = self.encoder.encode_point(point)
                        counter += 1
                    game_state = game_state.apply_move(move)
                    first_move_done = True

        feature_file_base = self.train_data_dir + '/' + data_file_name + '_features_%d'
        label_file_base = self.train_data_dir + '/' + data_file_name + '_labels_%d'

        # Due to files with large content, split up after chunksize
        chunk = 0
        chunksize = 1024
        while features.shape[0] > 0:
            feature_file = feature_file_base % chunk
            label_file = label_file_base % chunk
            chunk += 1
            current_features, features = features[:chunksize], features[chunksize:]
            current_labels, labels = labels[:chunksize], labels[chunksize:]  # <2>
            np.save(feature_file, current_features)
            np.save(label_file, current_labels)

    def consolidate_games(self, data_type, samples):
        files_needed = set(file_name for file_name, index in samples)
        file_names = []
        for zip_file_name in files_needed:
            file_name = zip_file_name.replace('.tar.gz', '') + data_type
            file_names.append(file_name)

        feature_list = []
        label_list = []
        for file_name in file_names:
            file_prefix = file_name.replace('.tar.gz', '')
            base = self.train_data_dir + '/' + file_prefix + '_features_*.npy'
            for feature_file in glob.glob(base):
                label_file = feature_file.replace('features', 'labels')
                x = np.load(feature_file)
                y = np.load(label_file)
                x = x.astype('float32')
                y = to_categorical(y.astype(int), 19 * 19)
                feature_list.append(x)
                label_list.append(y)
        features = np.concatenate(feature_list, axis=0)
        labels = np.concatenate(label_list, axis=0)
        np.save('{}/features_{}.npy'.format(self.train_data_dir, data_type), features)
        np.save('{}/labels_{}.npy'.format(self.train_data_dir, data_type), labels)

        return features, labels

    @staticmethod
    def get_handicap(sgf):
        go_board = Board(19, 19)
        first_move_done = False
        move = None
        game_state = GameState.new_game(19)
        if sgf.get_handicap() is not None and sgf.get_handicap() != 0:
            for setup in sgf.get_root().get_setup_stones():
                for move in setup:
                    row, col = move
                    go_board.place_stone(Player.black, Point(row + 1, col + 1))
            first_move_done = True
            game_state = GameState(go_board, Player.white, None, move)
        return game_state, first_move_done

    def num_total_examples(self, zip_file, game_list, name_list):
        total_examples = 0
        for index in game_list:
            name = name_list[index + 1]
            if name.endswith('.sgf'):
                sgf_content = zip_file.extractfile(name).read()
                sgf = Sgf_game.from_string(sgf_content)
                game_state, first_move_done = self.get_handicap(sgf)

                num_moves = 0
                for item in sgf.main_sequence_iter():
                    color, move = item.get_move()
                    if color is not None:
                        if first_move_done:
                            num_moves += 1
                        first_move_done = True
                total_examples = total_examples + num_moves
            else:
                raise ValueError(name + ' is not a valid sgf')
        return total_examples

    def map_to_workers(self, data_type, samples):
        """
        Intermediate method that distribute game extraction from zipped files work to workers
        Args:
            data_type: 'train' or 'test'
            samples: A list of tuples (file_name, game_index)
        """
        # Create a mapping of each filename to a list of game_index
        zip_names = set()
        indices_by_zip_name = {}
        for filename, index in samples:
            zip_names.add(filename)
            if filename not in indices_by_zip_name:
                indices_by_zip_name[filename] = []
            indices_by_zip_name[filename].append(index)

        # prepare a list of jobs
        zips_to_process = []
        for zip_name in zip_names:
            base_name = zip_name.replace('.tar.gz', '')
            data_file_name = base_name + data_type
            zips_to_process.append(
                (
                    self.__class__,
                    self.raw_data_dir,
                    self.exp_dir,
                    self.train_data_dir,
                    self.encoder_string,
                    zip_name,
                    data_file_name,
                    indices_by_zip_name[zip_name]
                )
            )

        cores = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=cores)
        p = pool.map_async(worker, zips_to_process)
        try:
            _ = p.get()
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            sys.exit(-1)
