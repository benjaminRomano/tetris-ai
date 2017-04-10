from collections import deque

import keras
import numpy as np
from keras.layers import Dense, Flatten, Input, merge
from keras.layers.convolutional import Conv2D
from keras.models import Model, Sequential
from keras.optimizers import Adam

import tetris as game
from replay import ReplayMemory

num_actions = 6
state_space_shape = (6,10)
observation_frames_length = 10

model = Sequential()
model.add(Conv2D(16, 2, activation='relu', input_shape=(state_space_shape[0], state_space_shape[1], observation_frames_length)))

model.add(Conv2D(32, (2,2), activation='relu'))

model.add(Conv2D(64, (2,2), activation='relu'))

model.add(Flatten())

print model.output_shape

model.add(Dense(87))
model.add(Dense(num_actions))

model.compile(loss='mean_squared_logarithmic_error',
    optimizer=Adam(lr=0.0003),
    metrics=['accuracy'])

replay_memory = ReplayMemory(model, discount=1.0)

def epsilon(episode):
    return max(1.0 - ((episode - 20) / 40.0), 0.2)

def numBatches(totalMoves):
    # return max(1, int(math.floor(math.log(totalMoves / 32, 2))))
    return 5

total_moves = 1
episode = 0
moving_num_moves = np.zeros((100,), dtype='float32')

# # TODO: Initialize game here
game_state = game.GameState()

while True:
    board, _, _ = game_state.frame_step(0)
    board = np.asarray(board, dtype='int')
    state_t1 = np.stack([board for _ in range(observation_frames_length)], axis = 2)

    total_loss = 0
    total_acc = 0
    game_over = False
    game_over_count = 0
    num_moves = 1

    while not game_over:
        action_t1 = np.random.randint(0, num_actions, 1)[0] \
            if np.random.rand() <= epsilon(episode) \
            else np.argmax(model.predict(np.expand_dims(state_t1, axis=0)))

        board, reward_t1, game_over = game_state.frame_step(action_t1)
        board = np.asarray(board, dtype='int')
        board = np.reshape(board, (state_space_shape[0], state_space_shape[1], 1))
        state_t2 = np.append(board, state_t1[:,:,0:observation_frames_length - 1], axis = 2)

        # reward_t1 = 0 if game_over else 1
        
        replay_memory.save(state_t1.copy(), action_t1, reward_t1, state_t2.copy(), game_over)

        state_t1 = state_t2

        if episode >= 10:
            loss = acc = 0
            batches = numBatches(total_moves)
            for batch in range(batches):
                xInputs, yOutputs = replay_memory.getBatch()
                l, a = model.train_on_batch(xInputs, yOutputs)
                loss += l
                acc += a
            
            total_loss += loss / batches
            total_acc += acc / batches

        num_moves += 1
        total_moves += 1

        if game_over:
            game_over_count += 1

    moving_num_moves = np.roll(moving_num_moves, -1, axis=0)
    moving_num_moves[-1] = num_moves
    moving_average = np.sum(moving_num_moves) / 100.0

    episode += 1
    
    print 'Episode: {} | Moves: {} | Moving Average: {} | Epsilon: {} | Loss: {} | Acc: {} | Total Lines: {}'\
        .format(episode, num_moves, moving_average, epsilon(episode), total_loss / num_moves, total_acc / num_moves, game_state.total_lines)

    model.save_weights('tetris' + '.save.h5', overwrite=True)
