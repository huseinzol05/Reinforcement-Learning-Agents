import numpy as np
from ple import PLE
import random
from ple.games.flappybird import FlappyBird
from evolution_strategy import *
import pickle

class Model:
    def __init__(self, input_size, layer_size, output_size):
        self.weights = [np.random.randn(input_size, layer_size),
                        np.random.randn(layer_size, output_size),
                        np.random.randn(1, layer_size)]

    def predict(self, inputs):
        out = np.dot(inputs, self.weights[0]) + self.weights[-1]
        out = np.dot(out, self.weights[1])
        return out

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights

class Agent:

    POPULATION_SIZE = 15
    SIGMA = 0.1
    LEARNING_RATE = 0.03
    GAMMA = 0.99
    # based on documentation, features got 8 dimensions
    OUTPUT_SIZE = 2
    # output is 2 dimensions, 0 = do nothing, 1 = jump

    def __init__(self, model, screen=False, forcefps=True):
        self.model = model
        self.game = FlappyBird(pipe_gap=125)
        self.env = PLE(self.game, fps=30, display_screen=screen, force_fps=forcefps)
        self.env.init()
        self.env.getGameState = self.game.getGameState
        self.es = Deep_Evolution_Strategy(self.model.get_weights(), self.get_reward, self.POPULATION_SIZE, self.SIGMA, self.LEARNING_RATE)
        self.rewards = []

    def get_predicted_action(self, sequence):
        prediction = self.model.predict(np.array(sequence))[0]
        return np.random.choice(self.OUTPUT_SIZE,p=prediction)

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

    def discount_rewards(rewards):
        discounted_r = np.zeros_like(rewards)
        running_add = 0
        for t in range(rewards.shape[0]):
            running_add = running_add * self.GAMMA + rewards[t]
            discounted_r[t] = running_add
        return discounted_r

    def get_reward(self, weights):
        self.model.weights = weights
        total_reward = 0.0
        self.env.reset_game()
        state = self.get_state()
        sequence = [state]
        done = False
        ep_history = []
        while not done:
            action = self.get_predicted_action(sequence)
            real_action = 119 if action == 1 else None
            reward = self.env.act(real_action)
            reward += random.choice([0.0001, -0.0001])
            total_reward += reward
            next_state = self.get_state()
            ep_history.append([state,action,total_reward,next_state])
            state = next_state
            sequence = [state]
            done = self.env.game_over()
        ep_history = np.array(ep_history)
        ep_history[:,2] = discount_rewards(ep_history[:,2])
        predicted = self.model.predict(np.vstack(ep_history[:,0]))
        action_aranged = np.arange(0, predicted.shape[0]) * predicted.shape[1] + ep_history[:,1]
        responsible_outputs = predicted.reshape([-1])[action_aranged]
        loss = -np.mean(np.log(responsible_outputs)*ep_history[:,2])
        self.rewards.append(total_reward)
        return total_reward

    def fit(self, iterations, checkpoint):
        self.es.train(iterations,print_every=checkpoint)

    def play(self, debug=False, not_realtime=False):
        total_reward = 0.0
        current_reward = 0
        self.env.force_fps = not_realtime
        self.env.reset_game()
        state = self.get_state()
        sequence = [state]
        done = False
        while not done:
            action = self.get_predicted_action(sequence)
            real_action = 119 if action == 1 else None
            action_string = 'eh, jump!' if action == 1 else 'erm, do nothing..'
            if debug and total_reward > current_reward:
                print(action_string, 'total rewards:', total_reward)
            current_reward = total_reward
            total_reward += self.env.act(real_action)
            state = self.get_state()
            sequence = [state]
            done = self.env.game_over()
        print('game over!')

model = Model(8, 500, 2)
agent = Agent(model, screen=True, forcefps=True)
