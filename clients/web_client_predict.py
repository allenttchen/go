import os

import h5py
from keras.models import load_model
from dlgo.agent.predict import DeepLearningAgent, load_prediction_agent
from dlgo.api.server import get_web_app
from dlgo.encoders.oneplane import OnePlaneEncoder
from constants import ROOT_DIR

# load model
# model_path = os.path.join(ROOT_DIR, "experiments/small_cnn_005/checkpoints/small_cnn_epoch_10.h5")
# model = load_model(model_path)
# go_board_rows, go_board_cols = 19, 19
# encoder = OnePlaneEncoder((go_board_rows, go_board_cols))
# dl_agent = DeepLearningAgent(model, encoder)

# load agent
agent_file = h5py.File(os.path.join(ROOT_DIR, "bot/AI_medium_oneplane_3000_bot.h5"), "r")
agent = load_prediction_agent(agent_file)

# initialize web app
web_app = get_web_app({'predict': agent})
web_app.run()
