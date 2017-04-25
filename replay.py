# Written by Chris Caruso
# Released under a "MIT" license

import numpy as np

# Note: This is a direct implementation of Replay Memory from "Playing Atari with Deep Reinforcement Learning"
class ReplayMemory():

    def __init__(self, model, batchSize=32, stateToBatchTransform=None, maxSize=25000, discount=0.95):
        self.model = model
        self.batchSize = batchSize
        self.stateToBatchTransform = stateToBatchTransform
        self.maxSize = maxSize
        self.discount = discount
        self.experiences = np.zeros((self.maxSize, 5), dtype=object)
        self.count = 0

    def save(self, stateT1, actionT1, rewardT1, stateT2, gameOver):
        
        experience = self.experiences[self.count % self.maxSize]
        
        experience[0] = [stateT1] if type(self.model.input_shape) is tuple else stateT1
        experience[1] = actionT1
        experience[2] = rewardT1
        experience[3] = [stateT2] if type(self.model.input_shape) is tuple else stateT2
        experience[4] = gameOver

        self.count += 1

    def getBatch(self):

        numActions = self.model.output_shape[-1]

        # Inputs for each batch. Keras models can have more than one input 
        xInputs = [(self.batchSize,) + self.model.input_shape[1:]] \
            if type(self.model.input_shape) is tuple else \
            [(self.batchSize,) + shape[1:] for shape in self.model.input_shape]
        xInputs = [np.zeros(shape, dtype='float32') for shape in xInputs]

        # Target actions for input
        yTargets = np.zeros((self.batchSize, numActions))
                
        randomExperiences = np.random.choice(np.arange(min(self.count, self.maxSize)), self.batchSize, replace=self.count<self.batchSize)
        for i in range(self.batchSize):    
            
            stateT1, actionT1, rewardT1, stateT2, gameOver = self.experiences[randomExperiences[i]]
            
            for j in range(len(xInputs)):
                xInputs[j][i] = stateT1[j] if self.stateToBatchTransform is None else \
                    self.stateToBatchTransform(stateT1, j) if len(xInputs) > 1 else \
                        self.stateToBatchTransform(stateT1)
            
            inputT1 = np.array([xInputs[j][i] for j in range(len(xInputs))])

            yTargets[i] = self.model.predict(inputT1)[0] # yT1
            
            inputT2 = [(1,) + self.model.input_shape[1:]] \
                if type(self.model.input_shape) is tuple else \
                [(1,) + shape[1:] for shape in self.model.input_shape]
            
            inputT2 = [np.zeros(shape, dtype='float32') for shape in inputT2]

            for j in range(len(inputT2)):
                inputT2[j][0] = stateT2[j] if self.stateToBatchTransform is None else \
                    self.stateToBatchTransform(stateT2, j) if len(inputT2) > 1 else \
                        self.stateToBatchTransform(stateT2)

            # Predict output (actions) for the next state (stateT2),
            # and pick the output (action) with the highest probability
            #   ( compute argmax(Q(s`, a`)) )
            Qsa = np.max(self.model.predict(inputT2)[0])
            
            #                                                rewardT1 + gamma * argmax(Q(s`, a`))
            yTargets[i, actionT1] = rewardT1 if gameOver else rewardT1 + self.discount * Qsa
                                
        return xInputs, yTargets

# Note: This is an extension using "Accelerating Minibatch Stochastic Gradient Descent using Stratified Sampling"
class StratifiedReplayMemory():

    def __init__(self, model, numStratifications, batchSize=32, stateToBatchTransform=None, maxSize=25000, discount=0.95):
        self.model = model
        self.numStratifications = numStratifications
        self.batchSize = batchSize
        self.stratificationBatchSize = int(self.batchSize / self.numStratifications) 
        self.stateToBatchTransform = stateToBatchTransform
        self.maxSize = maxSize
        self.stratificationSize = int(self.maxSize / self.numStratifications)
        self.discount = discount
        self.experiences = np.zeros((self.numStratifications, self.stratificationSize, 5), dtype=object)
        self.count = np.zeros((self.numStratifications,), dtype='int32')

    def save(self, stateT1, actionT1, rewardT1, stateT2, gameOver, stratification):
        
        experience = self.experiences[stratification][self.count[stratification] % self.stratificationSize]
        
        experience[0] = [stateT1] if type(self.model.input_shape) is tuple else stateT1
        experience[1] = actionT1
        experience[2] = rewardT1
        experience[3] = [stateT2] if type(self.model.input_shape) is tuple else stateT2
        experience[4] = gameOver

        self.count[stratification] += 1

    def getBatch(self):

        numActions = self.model.output_shape[-1]
        computedBatchSize = self.stratificationBatchSize * self.numStratifications

        # Inputs for each batch. Keras models can have more than one input 
        xInputs = [(computedBatchSize,) + self.model.input_shape[1:]] \
            if type(self.model.input_shape) is tuple else \
            [(computedBatchSize,) + shape[1:] for shape in self.model.input_shape]
        xInputs = [np.zeros(shape, dtype='float32') for shape in xInputs]

        # Target actions for input
        yTargets = np.zeros((computedBatchSize, numActions))
        
        for s in range(self.numStratifications):

            randomExperiences = np.random.choice(np.arange(min(self.count[s], self.stratificationSize)), self.stratificationBatchSize, replace=self.count[s]<self.stratificationBatchSize)
            for i in range(self.stratificationBatchSize):

                stateT1, actionT1, rewardT1, stateT2, gameOver = self.experiences[s][randomExperiences[i]]

                i = s * self.stratificationBatchSize + i
                
                for j in range(len(xInputs)):
                    xInputs[j][i] = stateT1[j] if self.stateToBatchTransform is None else \
                        self.stateToBatchTransform(stateT1, j) if len(xInputs) > 1 else \
                            self.stateToBatchTransform(stateT1)
                
                inputT1 = np.array([xInputs[j][i] for j in range(len(xInputs))])

                yTargets[i] = self.model.predict(inputT1)[0] # yT1
                
                inputT2 = [(1,) + self.model.input_shape[1:]] \
                    if type(self.model.input_shape) is tuple else \
                    [(1,) + shape[1:] for shape in self.model.input_shape]
                
                inputT2 = [np.zeros(shape, dtype='float32') for shape in inputT2]

                for j in range(len(inputT2)):
                    inputT2[j][0] = stateT2[j] if self.stateToBatchTransform is None else \
                        self.stateToBatchTransform(stateT2, j) if len(inputT2) > 1 else \
                            self.stateToBatchTransform(stateT2)

                # Predict output (actions) for the next state (stateT2),
                # and pick the output (action) with the highest probability
                #   ( compute argmax(Q(s`, a`)) )
                Qsa = np.max(self.model.predict(inputT2)[0])
                
                #                                                 rewardT1 + gamma         * argmax(Q(s`, a`))
                yTargets[i, actionT1] = rewardT1 if gameOver else rewardT1 + self.discount * Qsa
        
        shuffle = np.arange(computedBatchSize)
        np.random.shuffle(shuffle)

        xInputs = [xInput[shuffle] for xInput in xInputs]
        yTargets = yTargets[shuffle]

        return xInputs, yTargets
