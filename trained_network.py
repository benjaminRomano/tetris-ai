from collections import deque

import keras
import numpy as np
from keras.layers import Dense, Flatten, Input, concatenate
from keras.layers.convolutional import Conv2D
from keras.models import Model, Sequential
from keras.optimizers import Adam

import reduced_tetris as game
from replay import StratifiedReplayMemory

num_actions = 6
state_space_shape = (6,10)
observation_frames_length = 10

xState = Input(shape=(state_space_shape[0], state_space_shape[1], observation_frames_length))

conv1 = Conv2D(128, (2,2), activation='relu')(xState)
conv2 = Conv2D(256, (2,2), activation='relu')(conv1)

fState = Flatten()(xState)
fConvState = Flatten()(conv2)

a = concatenate([fState, fConvState])

b = Dense(256, activation='relu')(a)
b = Dense(128, activation='relu')(b)

c = Dense(256, activation='relu')(a)
c = Dense(128, activation='relu')(c)  

d = concatenate([b, c])

yOutcome = Dense(num_actions, activation='relu')(d)

model = Model(input=[xState], output=[yOutcome])

model.compile(loss='mean_squared_error',
    optimizer=Adam(lr=0.00001),
    metrics=['accuracy'])

model.load_weights('tetris.v1.1.save.good.h5')

game_state = game.GameState()

while True:
    board, _, _ = game_state.frame_step(0)
    board = np.asarray(board, dtype='int')
    state_t1 = np.stack([board for _ in range(observation_frames_length)], axis = 2)

    game_over = False

    while not game_over:
        action_t1 = np.argmax(model.predict(np.expand_dims(state_t1, axis=0)))

        board, _, game_over = game_state.frame_step(action_t1)
        board = np.asarray(board, dtype='int')
        board = np.reshape(board, (state_space_shape[0], state_space_shape[1], 1))
        state_t2 = np.append(board, state_t1[:,:,0:observation_frames_length - 1], axis = 2)
        state_t1 = state_t2