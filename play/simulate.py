import argparse
import datetime
from collections import namedtuple
import os

import h5py
from dlgo import agent
from dlgo import scoring
from dlgo import rl
from dlgo.goboard import GameState, Player, Point
from dlgo.agent.pg import load_policy_agent
from dlgo.rl.experience import ExperienceCollector, combine_experience
from constants import ROOT_DIR


BOARD_SIZE = 19
COLS = 'ABCDEFGHJKLMNOPQRST'
STONE_TO_CHAR = {
    None: '.',
    Player.black: 'x',
    Player.white: 'o',
}


def avg(items):
    if not items:
        return 0.0
    return sum(items) / float(len(items))


def print_board(board):
    for row in range(BOARD_SIZE, 0, -1):
        line = []
        for col in range(1, BOARD_SIZE + 1):
            stone = board.get(Point(row=row, col=col))
            line.append(STONE_TO_CHAR[stone])
        print('%2d %s' % (row, ''.join(line)))
    print('   ' + COLS[:BOARD_SIZE])


class GameRecord(namedtuple('GameRecord', 'moves winner margin')):
    pass


def name(player):
    if player == Player.black:
        return 'B'
    return 'W'


def simulate_game(black_player, white_player):
    """
    Simulate a game between two agents
    """
    # Configs
    moves = []
    game = GameState.new_game(BOARD_SIZE)
    agents = {
        Player.black: black_player,
        Player.white: white_player,
    }

    # simulate games
    while not game.is_over():
        next_move = agents[game.next_player].select_move(game)
        moves.append(next_move)
        game = game.apply_move(next_move)

    print_board(game.board)
    game_result = scoring.compute_game_result(game)
    print(game_result)

    return GameRecord(
        moves=moves,
        winner=game_result.winner,
        margin=game_result.winning_margin,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning-agent', required=True)
    parser.add_argument('--num-games', '-n', type=int, default=10)
    parser.add_argument('--game-log-out', required=True)
    parser.add_argument('--experience-out', required=True)
    parser.add_argument('--temperature', type=float, default=0.0)

    args = parser.parse_args()
    learning_agent_path = os.path.join(ROOT_DIR, "bot", args.learning_agent)
    game_log_path = os.path.join(ROOT_DIR, "play/experience", args.game_log_out)
    experience_out_path = os.path.join(ROOT_DIR, "play/experience", args.experience_out)

    # Initialize agents and experience collectors
    agent1 = load_policy_agent(h5py.File(learning_agent_path))
    agent2 = load_policy_agent(h5py.File(learning_agent_path))
    agent1.set_temperature(args.temperature)
    agent2.set_temperature(args.temperature)

    collector1 = ExperienceCollector()
    collector2 = ExperienceCollector()

    color1 = Player.black
    logf = open(game_log_path, 'a')
    logf.write('Begin training at %s\n' % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),))
    for i in range(args.num_games):
        print(f"Simulating game {i + 1}/{args.num_games}...")
        collector1.begin_episode()
        agent1.set_collector(collector1)
        collector2.begin_episode()
        agent2.set_collector(collector2)

        if color1 == Player.black:
            black_player, white_player = agent1, agent2
        else:
            white_player, black_player = agent2, agent1
        game_record = simulate_game(black_player, white_player)
        if game_record.winner == color1:
            print('Agent 1 wins.')
            collector1.complete_episode(reward=1)
            collector2.complete_episode(reward=-1)
        else:
            print('Agent 2 wins.')
            collector2.complete_episode(reward=1)
            collector1.complete_episode(reward=-1)
        # alternate between black and white for each agent
        color1 = color1.other

    experience = combine_experience([collector1, collector2])
    logf.write('Saving experience buffer to %s\n' % experience_out_path)
    with h5py.File(experience_out_path, 'w') as experience_outf:
        experience.serialize(experience_outf)


if __name__ == '__main__':
    main()
