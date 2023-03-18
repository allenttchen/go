import time

from dlgo.agent import naive
from dlgo import goboard_slow, gotypes
from dlgo.utils import print_move, print_board


def main():
    board_size = 9
    game = goboard_slow.GameState.new_game(board_size)
    bots = {
        gotypes.Player.black: naive.RandomBot(),
        gotypes.Player.white: naive.RandomBot(),
    }
    print_board(game.board)
    while not game.is_over():
        time.sleep(0.3)

        #print(chr(27) + "[2J")
        bot_move = bots[game.next_player].select_move(game)
        print_move(game.next_player, bot_move)
        game = game.apply_move(bot_move)
        print_board(game.board)


if __name__ == '__main__':
    main()