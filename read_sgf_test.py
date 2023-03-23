from dlgo.gosgf import Sgf_game
from dlgo.goboard import GameState, Move
from dlgo.gotypes import Point
from dlgo.utils import print_board, print_move
from dlgo.preprocessing.processor import GoDataProcessor
from dlgo.preprocessing.generator import DataGenerator

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
    processor = GoDataProcessor(
        index_page='./records/kgs/kgs_index.html',
        data_directory='./records/kgs/data'
    )
    dataGenerator = processor.load_go_data('train', 100, use_generator=True)
    print(dataGenerator.get_num_samples())
    generator = dataGenerator.generate(batch_size=64)
    features, labels = next(generator)
    print(features.shape)
    print(features[0])
    print(labels.shape)
    print(labels[0])
    # samples = [
    #     ('KGS-2001-19-2298-.tar.gz', 2),
    #     ('KGS-2001-19-2298-.tar.gz', 5),
    #     ('KGS-2002-19-3646-.tar.gz', 1),
    #     ('KGS-2002-19-3646-.tar.gz', 10),
    #     ('KGS-2002-19-3646-.tar.gz', 11),
    # ]
    # generator = DataGenerator(data_directory='./records/kgs/data', samples=samples)
    # next(generator.generate())
