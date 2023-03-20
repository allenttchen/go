import numpy as np
from keras.models import Sequential
from keras.layers import Dense


np.random.seed(123)
print()
X = np.load('../records/large_mcts/features-40k.npy')
Y = np.load('../records/large_mcts/labels-40k.npy')
samples = X.shape[0]
board_size = 9 * 9

X = X.reshape(samples, board_size)
Y = Y.reshape(samples, board_size)

train_samples = int(0.9 * samples)
X_train, X_test = X[:train_samples], X[train_samples:]
Y_train, Y_test = Y[:train_samples], Y[train_samples:]

model = Sequential()
model.add(Dense(1000, activation='sigmoid', input_shape=(board_size, )))
model.add(Dense(500, activation='sigmoid'))
model.add(Dense(board_size, activation='softmax'))
model.summary()

model.compile(
    loss='categorical_crossentropy',
    optimizer='sgd',
    metrics=['accuracy'],
)

model.fit(
    X_train,
    Y_train,
    epochs=15,
    verbose=1,
    validation_data=(X_test, Y_test),
)

score = model.evaluate(X_test, Y_test, verbose=0)
print(f'Test loss: {score[0]}')
print(f'Test accuracy: {score[1]}')

model.save('mlp_model')
