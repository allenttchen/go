import argparse
import os

import h5py

from dlgo import agent, api
from dlgo.agent.predict import DeepLearningAgent, load_prediction_agent
from constants import ROOT_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bind-address', default='127.0.0.1')
    parser.add_argument('--port', '-p', type=int, default=5000)
    parser.add_argument('--pg-agent')
    parser.add_argument('--predict-agent')
    parser.add_argument('--q-agent')
    parser.add_argument('--ac-agent')

    args = parser.parse_args()

    agent_file = h5py.File(os.path.join(ROOT_DIR, args.predict_agent), "r")
    agent = load_prediction_agent(agent_file)
    bots = {
        'predict': agent
    }
    web_app = api.get_web_app(bots)
    web_app.run(host=args.bind_address, port=args.port, threaded=False)


if __name__ == '__main__':
    main()