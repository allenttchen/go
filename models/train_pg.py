import argparse
import os

import h5py

from dlgo import agent, rl
from constants import ROOT_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning-agent', required=True)
    parser.add_argument('--agent-out', required=True)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--bs', type=int, default=512)
    parser.add_argument('experience_data', nargs='+')

    args = parser.parse_args()
    learning_agent_path = os.path.join(ROOT_DIR, args.learning_agent)

    learning_agent = agent.load_policy_agent(h5py.File(learning_agent_path))
    for exp_filename in args.experience_data:
        experience_data_path = os.path.join(ROOT_DIR, exp_filename)
        print('Training with %s...' % experience_data_path)
        exp_buffer = rl.load_experience(h5py.File(experience_data_path))
        learning_agent.train(exp_buffer, lr=args.lr, batch_size=args.bs)

    with h5py.File(args.agent_out, 'w') as updated_agent_outf:
        learning_agent.serialize(updated_agent_outf)


if __name__ == '__main__':
    main()
