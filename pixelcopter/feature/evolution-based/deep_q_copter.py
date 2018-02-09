import numpy as np
from ple import PLE
import random
from ple.games.pixelcopter import Pixelcopter
from collections import deque
from evolution_strategy import *

class Agent:

    MEMORY_SIZE = 10000
    BATCH = 32
    POPULATION_SIZE = 15
    SIGMA = 0.1
    LEARNING_RATE = 0.03
    EPSILON = 1
    DECAY_RATE = 0.005
    MIN_EPSILON = 0.1
    GAMMA = 0.99
    INPUT_SIZE = 7
    OUTPUT_SIZE = 2
    MEMORIES = deque()
    # based on documentation, features got 8 dimensions

    def __init__(self, model, screen=False, forcefps=True):
        self.model = model
        self.game = Pixelcopter(width=int(48 * 5), height=int(48 * 5))
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
        X = np.empty((replay_size, self.INPUT_SIZE))
        Y = np.empty((replay_size, self.OUTPUT_SIZE))
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
        dead = False
        while not dead:
            state = self.get_state()
            action  = self._select_action(state)
            real_action = 119 if action == 1 else None
            reward = self.env.act(real_action)
            new_state = self.get_state()
            dead = self.env.game_over()
            self._memorize(state, action, reward, new_state, dead)
        batch_size = min(len(self.MEMORIES), self.BATCH)
        replay = random.sample(self.MEMORIES, batch_size)
        X, Y = self._construct_memories(replay)
        actions = self.model.predict(X)
        return -np.mean(np.square(Y - actions))

    def fit(self, iterations, checkpoint):
        self.es.train(iterations,print_every=checkpoint)

    def play(self, debug=False, not_realtime=False):
        total_reward = 0.0
        current_reward = 0
        self.env.force_fps = not_realtime
        self.env.reset_game()
        done = False
        while not done:
            state = self.get_state()
            action = self.get_predicted_action([state])
            real_action = 119 if action == 1 else None
            action_string = 'eh, jump!' if action == 1 else 'erm, do nothing..'
            if debug and total_reward > current_reward:
                print(action_string, 'total rewards:', total_reward)
            current_reward = total_reward
            total_reward += self.env.act(real_action)
            done = self.env.game_over()
        print('game over!')
