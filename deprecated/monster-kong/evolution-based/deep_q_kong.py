import numpy as np
from ple import PLE
import random
from ple.games.monsterkong import MonsterKong
from collections import deque
from scipy.misc import imresize
from evolution_strategy import *

class Agent:

    MEMORY_SIZE = 300
    BATCH = 32
    POPULATION_SIZE = 15
    SIGMA = 0.1
    LEARNING_RATE = 0.03
    EPSILON = 1
    MIN_EPSILON = 0.1
    WATCHING = 10000
    FEATURES = 8
    GAMMA = 0.99
    MEMORIES = deque()
    INITIAL_IMAGES = np.zeros((80, 80, 4))
    # based on documentation, features got 8 dimensions

    def __init__(self, model, screen=False, forcefps=True):
        self.model = model
        self.game = MonsterKong()
        self.env = PLE(self.game, fps=30, display_screen=screen, force_fps=forcefps)
        self.env.init()
        self.env.getGameState = self.game.getGameState
        self.es = Deep_Evolution_Strategy(self.model.get_weights(), self.get_reward, self.POPULATION_SIZE, self.SIGMA, self.LEARNING_RATE)

    def _get_image(self, image):
        r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return imresize(gray, size = (80, 80))

    def _map_action(self, action):
        if action == 0:
            return 97
        if action == 1:
            return 100
        if action == 2:
            return 119
        if action == 3:
            return 115
        if action == 4:
            return 32

    def _memorize(self, state, action, reward, new_state, done):
        self.MEMORIES.append((state, action, reward, new_state, done))
        if len(self.MEMORIES) > self.MEMORY_SIZE:
            self.MEMORIES.popleft()

    def _construct_memories(self, replay):
        states = np.array([a[0] for a in replay])
        new_states = np.array([a[3] for a in replay])
        Q = self.model.predict(states)
        Q_new = self.model.predict(new_states)
        replay_size = len(replay)
        X = np.empty((replay_size, 80, 80, 4))
        Y = np.empty((replay_size, 5))
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
        if random.random() > self.EPSILON:
            prediction = np.argmax(self.model.predict(np.array(sequence))[0])
        else:
            prediction = np.random.randint(5)
        self.EPSILON -= (self.EPSILON / self.WATCHING)
        return prediction

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
        for i in range(self.INITIAL_IMAGES.shape[2]):
            self.INITIAL_IMAGES[:,:,i] = state
        dead = False
        while not dead:
            action = self.get_predicted_action([self.INITIAL_IMAGES])
            real_action = self._map_action(action)
            reward = self.env.act(real_action)
            reward += random.choice([0.0001, -0.0001])
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

    def play(self, debug=False, not_realtime=False):
        total_reward = 0.0
        current_reward = 0
        self.env.force_fps = not_realtime
        self.env.reset_game()
        state = self._get_image(self.env.getScreenRGB())
        for k in range(self.INITIAL_IMAGES.shape[2]):
            self.INITIAL_IMAGES[:,:,k] = state
        done = False
        while not done:
            state = self.get_state()
            action = np.argmax(self.predict(np.array([self.INITIAL_IMAGES]))[0])
            real_action = 119 if action == 1 else None
            action_string = 'eh, jump!' if action == 1 else 'erm, do nothing..'
            if debug and total_reward > current_reward:
                print(action_string, 'total rewards:', total_reward)
            current_reward = total_reward
            total_reward += self.env.act(real_action)
            done = self.env.game_over()
        print('game over!')
