import time

from dlgo import agent, mcts
from dlgo import goboard, gotypes
from dlgo.utils import print_move, print_board


def main():
    board_size = 9
    game = goboard.GameState.new_game(board_size)
    bots = {
        gotypes.Player.black: agent.RandomBot(),
        gotypes.Player.white: agent.RandomBot(),
        #gotypes.Player.white: mcts.MCTSAgent(num_rounds=10, temperature=1.5),
    }
    print_board(game.board)
    while not game.is_over():
        time.sleep(0.3)

        #print(chr(27) + "[2J")
        bot_move = bots[game.next_player].select_move(game)
        print_move(game.next_player, bot_move)
        game = game.apply_move(bot_move)
        print_board(game.board)
    print(game.winner())


if __name__ == '__main__':
    main()