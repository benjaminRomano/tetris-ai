
import gym, keras, gc, math
import numpy as np
from keras.layers import Dense, Input, merge, Flatten
from keras.models import Model
from keras.optimizers import Adam
from replay import StratifiedReplayMemory

env = gym.make('CartPole-v0')
numActions = env.action_space.n
observationSpaceShape = env.observation_space.shape[0] + 1
maxRememberenceLength = 10

xStates = Input(shape=(maxRememberenceLength, observationSpaceShape))
a = Flatten()(xStates)

b = Dense(256, activation='relu')(a)
b = Dense(128, activation='relu')(b)

c = Dense(256, activation='relu')(a)
c = Dense(128, activation='relu')(c)

d = merge([b, c], mode='concat')

yOutcome = Dense(2, activation='relu')(d)

model = Model(input=[xStates], output=[yOutcome])

model.compile(loss='mean_squared_logarithmic_error',
    optimizer=Adam(lr=0.0003),
    metrics=['accuracy'])

replayMemory = StratifiedReplayMemory(model, 2, batchSize=32, discount=1.0)

def epsilon(episode):
    return max(1.0 - ((episode - 20) / 40.0), 0.2)
    do_nothing = np.zeros(num_actions)
def numBatches(totalMoves):
    # return max(1, int(math.floor(math.log(totalMoves / 32, 2))))
    return 5

totalMoves = 1
episode = 0
movingNumMoves = np.zeros((100,), dtype='float32')
env.monitor.start('./tmp/cartpole', force=True)

while np.sum(movingNumMoves) / 100.0 < 195:

    states = np.zeros((maxRememberenceLength, observationSpaceShape), dtype='float32')
    states[0] = np.concatenate((env.reset(), [-1.0]))

    totalLoss = 0
    totalAcc = 0
    gameOver = False
    gameOverCount = 0
    numMoves = 1
    while not gameOver:

        env.render()

        stateT1 = states.copy()
        if numMoves >= maxRememberenceLength:
            states = np.roll(states, -1, axis=0)

        actionT1 = np.random.randint(0, numActions, 1)[0] \
            if np.random.rand() <= epsilon(episode) \
            else np.argmax(model.predict(stateT1.reshape((1,) + stateT1.shape)))

        stateT2, rewardT1, gameOver, info = env.step(actionT1)
        states[min(numMoves, maxRememberenceLength - 1)] = np.concatenate((stateT2, [actionT1]))
        stateT2 = states.copy()
        
        rewardT1 = 0 if gameOver else 1
        
        replayMemory.save(stateT1, actionT1, rewardT1, stateT2, gameOver, rewardT1)

        if episode >= 10:
            loss = acc = 0
            batches = numBatches(totalMoves)
            for batch in range(batches):
                xInputs, yOutputs = replayMemory.getBatch()
                l, a = model.train_on_batch(xInputs, yOutputs)
                loss += l
                acc += a
            
            totalLoss += loss / batches
            totalAcc += acc / batches

        numMoves += 1
        totalMoves += 1

        if gameOver:
            gameOverCount += 1

    movingNumMoves = np.roll(movingNumMoves, -1, axis=0)
    movingNumMoves[-1] = numMoves
    movingAverage = np.sum(movingNumMoves) / 100.0

    episode += 1
    
    print 'Episode: {} | Moves: {} | Moving Average: {} | Epsilon: {} | Loss: {} | Acc: {} '\
        .format(episode, numMoves, movingAverage, epsilon(episode), totalLoss / numMoves, totalAcc / numMoves)

    model.save_weights('7' + '.save.h5', overwrite=True)

env.monitor.close()