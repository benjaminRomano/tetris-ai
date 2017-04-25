from collections import deque

import keras
import numpy as np
from keras.layers import Dense, Flatten, Input, concatenate
from keras.layers.convolutional import Conv2D
from keras.models import Model, Sequential
from keras.optimizers import Adam

from hueristic_ai import HueristicAI
import tetris as game
from replay import StratifiedReplayMemory

num_actions = 6
state_space_shape = (6,10)
observation_frames_length = 10
numSeedGames = 1000
num_hueristic_seeds = 5

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

# model.load_weights('tetris.v1.e6300.ma72.a665.tl670.save.h5')
# model.load_weights('tetris.v1.1.e22990.save.h5')
# model.load_weights('tetris.v1.1.save.good.h5')

# replay_memory = ReplayMemory(model, discount=1.0)
replay_memory = StratifiedReplayMemory(model, 3, discount=1.0)

def epsilon(episode):
    return max(1.0 - (episode / float(numSeedGames)), 0.05)

def numBatches(totalMoves):
    # return max(1, int(math.floor(math.log(totalMoves / 32, 2))))
    return 1

total_moves = 1
episode = 0
moving_num_moves = np.zeros((100,), dtype='float32')

# # TODO: Initialize game here
game_state = game.GameState()

ai = HueristicAI(game_state, -0.153, 0.605, -0.645, -0.442)

while True:
    board, _, _ = game_state.frame_step(0)
    board = np.asarray(board, dtype='int')
    state_t1 = np.stack([board for _ in range(observation_frames_length)], axis = 2)

    total_loss = 0
    total_acc = 0
    game_over = False
    game_over_count = 0
    num_moves = 1

    if episode < num_hueristic_seeds:

        while True:
            action_sequence = []

            if game_state.fallingPiece:
                best_piece, best_actions, score = ai.get_piece_move(game_state.board, [game_state.fallingPiece])

                action_sequence = best_actions
            else:
                action_sequence = [0]

            is_game_over = False

            for action in action_sequence:
                board, reward_t1, game_over = game_state.frame_step(action)
                board = np.asarray(board, dtype='int')
                board = np.reshape(board, (state_space_shape[0], state_space_shape[1], 1))
                state_t2 = np.append(board, state_t1[:,:,0:observation_frames_length - 1], axis = 2)

                stratification = 0 if reward_t1 < 0 else 1 if reward_t1 == 0 else 2
                
                replay_memory.save(state_t1.copy(), action, reward_t1, state_t2.copy(), game_over, stratification)

                state_t1 = state_t2

                if game_over:
                    is_game_over = True
                    break

            if is_game_over:
                break

        epiosde += 1
        continue

    while not game_over:
        action_t1 = np.random.randint(0, num_actions, 1)[0] \
            if np.random.rand() <= epsilon(episode) \
            else np.argmax(model.predict(np.expand_dims(state_t1, axis=0)))

        board, reward_t1, game_over = game_state.frame_step(action_t1)
        board = np.asarray(board, dtype='int')
        board = np.reshape(board, (state_space_shape[0], state_space_shape[1], 1))
        state_t2 = np.append(board, state_t1[:,:,0:observation_frames_length - 1], axis = 2)
        # print reward_t1
        # reward_t1 = 0 if game_over else 1'

        stratification = 0 if reward_t1 < 0 else 1 if reward_t1 == 0 else 2
        
        replay_memory.save(state_t1.copy(), action_t1, reward_t1, state_t2.copy(), game_over, stratification)

        state_t1 = state_t2

        if episode - num_hueristic_seeds >= numSeedGames:
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