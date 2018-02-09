import numpy as np
from ple import PLE
import random
from ple.games.pixelcopter import Pixelcopter
from scipy.misc import imresize
from scipy.misc import imsave
from evolution_strategy import *
import pickle

class Agent:

    POPULATION_SIZE = 15
    SIGMA = 0.1
    LEARNING_RATE = 0.003
    INITIAL_IMAGES = np.zeros((80, 80, 4))
    # based on documentation, features got 8 dimensions
    # output is 2 dimensions, 0 = do nothing, 1 = jump

    def __init__(self, model, screen=False, forcefps=True):
        self.model = model
        self.game = Pixelcopter(width=int(48 * 5), height=int(48 * 5))
        self.env = PLE(self.game, fps=30, display_screen=screen, force_fps=forcefps)
        self.env.init()
        self.env.getGameState = self.game.getGameState
        self.es = Deep_Evolution_Strategy(self.model.get_weights(), self.get_reward, self.POPULATION_SIZE, self.SIGMA, self.LEARNING_RATE)
        self.rewards = []

    def get_predicted_action(self, sequence):
        prediction = self.model.predict(np.array(sequence))[0]
        return np.argmax(prediction)

    def _get_image(self, image):
        r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return imresize(gray, size = (80, 80))

    def save(self, checkpoint_name):
        with open('%s-weight.p'%(checkpoint_name), 'wb') as fopen:
            pickle.dump(self.model.get_weights(), fopen)
        with open('%s-acc.p'%(checkpoint_name), 'wb') as fopen:
            pickle.dump(self.rewards, fopen)

    def load(self, checkpoint_name):
        with open('%s-weight.p'%(checkpoint_name), 'rb') as fopen:
            self.model.set_weights(pickle.load(fopen))
        with open('%s-acc.p'%(checkpoint_name), 'rb') as fopen:
            self.rewards = pickle.load(fopen)

    def get_state(self):
        state = self.env.getGameState()
        return np.array(list(state.values()))

    def get_reward(self, weights):
        self.model.set_weights(weights)
        total_reward = 0.0
        self.env.reset_game()
        state = self._get_image(self.env.getScreenRGB())
        imsave('state.png', state)
        for i in range(self.INITIAL_IMAGES.shape[2]):
            self.INITIAL_IMAGES[:,:,i] = state
        sequence = [self.INITIAL_IMAGES]
        done = False
        while not done:
            action = self.get_predicted_action(sequence)
            real_action = 119 if action == 1 else None
            reward = self.env.act(real_action)
            reward += random.choice([0.0001, -0.0001])
            total_reward += reward
            state = self._get_image(self.env.getScreenRGB())
            self.INITIAL_IMAGES = np.append(state.reshape([80, 80, 1]), self.INITIAL_IMAGES[:, :, :3], axis = 2)
            sequence = [self.INITIAL_IMAGES]
            done = self.env.game_over()
        self.rewards.append(total_reward)
        return total_reward

    def fit(self, iterations, checkpoint):
        self.es.train(iterations,print_every=checkpoint)

    def play(self, debug=False, not_realtime=False):
        total_reward = 0.0
        current_reward = 0
        self.env.force_fps = not_realtime
        self.env.reset_game()
        state = self._get_image(self.env.getScreenRGB())
        for i in range(self.INITIAL_IMAGES.shape[2]):
            self.INITIAL_IMAGES[:,:,i] = state
        sequence = [self.INITIAL_IMAGES]
        done = False
        while not done:
            action = self.get_predicted_action(sequence)
            real_action = 119 if action == 1 else None
            action_string = 'eh, jump!' if action == 1 else 'erm, do nothing..'
            if debug and total_reward > current_reward:
                print(action_string, 'total rewards:', total_reward)
            current_reward = total_reward
            total_reward += self.env.act(real_action)
            state = self._get_image(self.env.getScreenRGB())
            self.INITIAL_IMAGES = np.append(state.reshape([80, 80, 1]), self.INITIAL_IMAGES[:, :, :3], axis = 2)
            sequence = [self.INITIAL_IMAGES]
            done = self.env.game_over()
        print('game over!')
