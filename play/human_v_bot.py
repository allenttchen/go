from dlgo import goboard, gotypes
from dlgo.agent import naive
from dlgo.utils import print_board, print_move, point_from_coords
from six.moves import input


def main():
    board_size = 9
    game = goboard.GameState.new_game(board_size)
    bot = naive.RandomBot()
    print_board(game.board)

    while not game.is_over():

        if game.next_player == gotypes.Player.black:
            human_move = input('-- ')
            point = point_from_coords(human_move.strip())
            move = goboard.Move.play(point)
        else:
            move = bot.select_move(game)
        print_move(game.next_player, move)
        game = game.apply_move(move)
        print_board(game.board)


if __name__ == '__main__':
    main()
