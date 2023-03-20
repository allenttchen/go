import numpy as np
from keras import models


model = models.load_model('mlp_model')
test_board = np.array([[
    0, 0,  0,  0,  0, 0, 0, 0, 0,
    0, 0,  0,  0,  0, 0, 0, 0, 0,
    0, 0,  0,  0,  0, 0, 0, 0, 0,
    0, 1, -1,  1, -1, 0, 0, 0, 0,
    0, 1, -1,  1, -1, 0, 0, 0, 0,
    0, 0,  1, -1,  0, 0, 0, 0, 0,
    0, 0,  0,  0,  0, 0, 0, 0, 0,
    0, 0,  0,  0,  0, 0, 0, 0, 0,
    0, 0,  0,  0,  0, 0, 0, 0, 0,
]])
move_probs = model.predict(test_board)[0]
move_probs = move_probs.reshape((9, 9))
move_probs = np.around(move_probs, 3)
print(type(move_probs))
print(move_probs)
