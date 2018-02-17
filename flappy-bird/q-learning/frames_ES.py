import numpy as np
from ple import PLE
import random
from ple.games.flappybird import FlappyBird
from collections import deque
from evolution_strategy import *

class Model:
    def __init__(self):
        self.model = Sequential()
        self.model.add(Conv2D(16, (8, 8), strides = 4, padding='same',input_shape=(80, 80, 4)))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(32, (4, 4), strides = 2, padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(200))
        self.model.add(Dense(2))
        self.weights = self.model.get_weights()

    def predict(self, inputs):
        return self.model.predict(inputs)

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.model.set_weights(weights)

class Agent:

    MEMORY_SIZE = 300
    BATCH = 32
    POPULATION_SIZE = 15
    SIGMA = 0.1
    LEARNING_RATE = 0.03
    EPSILON = 1
    DECAY_RATE = 0.005
    MIN_EPSILON = 0.1
    FEATURES = 8
    GAMMA = 0.99
    INPUT_SIZE = 8
    OUTPUT_SIZE = 2
    MEMORIES = deque()
    INITIAL_IMAGES = np.zeros((80, 80, 4))
    # based on documentation, features got 8 dimensions

    def __init__(self, model, screen=False, forcefps=True):
        self.model = model
        self.game = FlappyBird(pipe_gap=125)
        self.env = PLE(self.game, fps=30, display_screen=screen, force_fps=forcefps)
        self.env.init()
        self.env.getGameState = self.game.getGameState
        self.es = Deep_Evolution_Strategy(self.model.get_weights(), self.get_reward, self.POPULATION_SIZE, self.SIGMA, self.LEARNING_RATE)

    def _memorize(self, state, action, reward, new_state, done):
        self.MEMORIES.append((state, action, reward, new_state, done))
        if len(self.MEMORIES) > self.MEMORY_SIZE:
            self.MEMORIES.popleft()

    def _select_action(self, state):
        if np.random.rand() < self.EPSILON:
            action = np.random.randint(2)
        else:
            action = self.get_predicted_action([state])
        return action

    def _construct_memories(self, replay):
        states = np.array([a[0] for a in replay])
        new_states = np.array([a[3] for a in replay])
        Q = self.model.predict(states)
        Q_new = self.model.predict(new_states)
        replay_size = len(replay)
        X = np.empty((replay_size, 8))
        Y = np.empty((replay_size, 2))
        for i in range(replay_size):
            state_r, action_r, reward_r, new_state_r, done_r = replay[i]
            target = Q[i]
            target[action_r] = reward_r
            if not done_r:
                target[action_r] += self.GAMMA * np.amax(Q_new[i])
            X[i] = state_r
            Y[i] = target
        return X, Y

    def get_predicted_action(self, sequence):
        prediction = self.model.predict(np.array(sequence))[0]
        return np.argmax(prediction)

    def get_state(self):
        state = self.env.getGameState()
        return np.array(list(state.values()))

    def save(self, checkpoint_name):
        with open('%s-weight.p'%(checkpoint_name), 'wb') as fopen:
            pickle.dump(self.model.get_weights(), fopen)

    def load(self, checkpoint_name):
        with open('%s-weight.p'%(checkpoint_name), 'rb') as fopen:
            self.model.set_weights(pickle.load(fopen))

    def get_reward(self, weights):
        self.model.set_weights(weights)
        self.env.reset_game()
        state = self._get_image(self.env.getScreenRGB())
        for k in range(self.INITIAL_IMAGES.shape[2]):
            self.INITIAL_IMAGES[:,:,k] = state
        dead = False
        while not dead:
            action  = self._select_action(self.INITIAL_IMAGES)
            real_action = 119 if action == 1 else None
            reward = self.env.act(real_action)
            state = self._get_image(self.env.getScreenRGB())
            new_state = np.append(state.reshape([80, 80, 1]), self.INITIAL_IMAGES[:, :, :3], axis = 2)
            dead = self.env.game_over()
            self._memorize(self.INITIAL_IMAGES, action, reward, new_state, dead)
            self.INITIAL_IMAGES = new_state
        batch_size = min(len(self.MEMORIES), self.BATCH)
        replay = random.sample(self.MEMORIES, batch_size)
        X, Y = self._construct_memories(replay)
        actions = self.model.predict(X)
        return -np.mean(np.square(Y - actions))

    def fit(self, iterations, checkpoint):
        self.es.train(iterations,print_every=checkpoint)

model = Model(8, 500, 2)
agent = Agent(model, screen=True, forcefps=True)
