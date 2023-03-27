import os

from dlgo.gosgf import Sgf_game
from dlgo.goboard import GameState, Move
from dlgo.gotypes import Point
from dlgo.utils import print_board, print_move
from dlgo.preprocessing.processor import GoDataProcessor
from dlgo.preprocessing.generator import DataGenerator
from dlgo.utils import print_board_from_lists, transform_move_vec_to_coords
from constants import ROOT_DIR

# sgf_content = "(;GM[1]FF[4]SZ[9];B[ee];W[ef];B[ff]" + \
#               ";W[df];B[fe];W[fc];B[ec];W[gd];B[fb])"
#
# sgf_game = Sgf_game.from_string(sgf_content)
#
# game_state = GameState.new_game(19)
#
# for item in sgf_game.main_sequence_iter():
#     print(item)
#     color, move_tuple = item.get_move()
#     if color is not None and move_tuple is not None:
#         row, col = move_tuple
#         point = Point(row + 1, col + 1)
#         move = Move.play(point)
#         print_move(color, move)
#         game_state = game_state.apply_move(move)
#         print_board(game_state.board)

if __name__ == '__main__':
    exp_dir = os.path.join(ROOT_DIR, "experiments/testing")
    data_dir = os.path.join(ROOT_DIR, "records/kgs/data")
    processor = GoDataProcessor(
        exp_dir=exp_dir,
        data_dir=data_dir,
        seed=123,
    )
    train_generator = processor.load_go_data('train', 100, use_generator=True)
    print(f"Total number of training moves: {train_generator.get_num_samples()}")
    generator = train_generator.generate(batch_size=64)
    features, labels = next(generator)
    print("first move: ")
    print(features.shape)
    print_board_from_lists(features[0][0])
    print(labels.shape)
    print(transform_move_vec_to_coords(labels[0], 19))

    # samples = [
    #     ('KGS-2001-19-2298-.tar.gz', 2),
    #     ('KGS-2001-19-2298-.tar.gz', 5),
    #     ('KGS-2002-19-3646-.tar.gz', 1),
    #     ('KGS-2002-19-3646-.tar.gz', 10),
    #     ('KGS-2002-19-3646-.tar.gz', 11),
    # ]
    # generator = DataGenerator(data_directory='./records/kgs/data', samples=samples)
    # next(generator.generate())
