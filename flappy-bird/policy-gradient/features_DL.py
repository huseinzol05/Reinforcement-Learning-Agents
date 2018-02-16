import numpy as np
from ple import PLE
import random
from ple.games.flappybird import FlappyBird
import tensorflow as tf
from collections import deque
import os
import pickle

class Agent:

    LEARNING_RATE = 0.003
    EPISODE = 500
    LAYER_SIZE = 500
    EPSILON = 1
    DECAY_RATE = 0.005
    MIN_EPSILON = 0.1
    GAMMA = 0.99
    INPUT_SIZE = 8
    # based on documentation, features got 8 dimensions
    OUTPUT_SIZE = 2
    # output is 2 dimensions, 0 = do nothing, 1 = jump

    def __init__(self, screen=False, forcefps=True):
        self.game = FlappyBird(pipe_gap=125)
        self.env = PLE(self.game, fps=30, display_screen=screen, force_fps=forcefps)
        self.env.init()
        self.env.getGameState = self.game.getGameState
        self.X = tf.placeholder(tf.float32, (None, self.INPUT_SIZE))
        self.REWARDS = tf.placeholder(tf.float32, (None))
        self.ACTIONS = tf.placeholder(tf.int32, (None))
        input_layer = tf.Variable(tf.random_normal([self.INPUT_SIZE, self.LAYER_SIZE]))
        bias = tf.Variable(tf.random_normal([self.LAYER_SIZE]))
        output_layer = tf.Variable(tf.random_normal([self.LAYER_SIZE, self.OUTPUT_SIZE]))
        feed_forward = tf.nn.relu(tf.matmul(self.X, input_layer) + bias)
        self.logits = tf.nn.softmax(tf.matmul(feed_forward, output_layer))
        indexes = tf.range(0, tf.shape(self.logits)[0]) * tf.shape(self.logits)[1] + self.ACTIONS
        responsible_outputs = tf.gather(tf.reshape(self.logits, [-1]), indexes)
        self.cost = -tf.reduce_mean(tf.log(responsible_outputs)*self.REWARDS)
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.LEARNING_RATE).minimize(self.cost)
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(tf.global_variables())
        self.rewards = []

    def predict(self, inputs):
        return self.sess.run(self.logits, feed_dict={self.X:inputs})

    def save(self, checkpoint_name):
        self.saver.save(self.sess, os.getcwd() + "/%s.ckpt" %(checkpoint_name))
        with open('%s-acc.p'%(checkpoint_name), 'wb') as fopen:
            pickle.dump(self.rewards, fopen)

    def load(self, checkpoint_name):
        self.saver.restore(self.sess, os.getcwd() + "/%s.ckpt" %(checkpoint_name))
        with open('%s-acc.p'%(checkpoint_name), 'rb') as fopen:
            self.rewards = pickle.load(fopen)

    def get_predicted_action(self, sequence):
        prediction = self.predict(np.array(sequence))[0]
        return np.argmax(prediction)

    def get_state(self):
        state = self.env.getGameState()
        return np.array(list(state.values()))

    def get_reward(self, iterations, checkpoint):
        for i in range(iterations):
            ep_history = []
            for k in range(self.EPISODE):
                total_reward = 0
                self.env.reset_game()
                done = False
                state = self.get_state()
                sequence = [state]
                while not done:
                    action  = self._select_action(state)
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
                sess.run(self.optimizer, feed_dict={self.X:np.vstack(ep_history[:,0]),
                                                    self.REWARDS:ep_history[:,2],
                                                    self.ACTIONS:ep_history[:,1]})
            self.rewards.append(total_reward)
            if (i+1) % checkpoint == 0:
                print('epoch:', i + 1, 'total rewards:', total_reward)
                print('epoch:', i + 1, 'cost:', cost)

    def fit(self, iterations, checkpoint):
        self.get_reward(iterations, checkpoint)

    def play(self, debug=False, not_realtime=False):
        total_reward = 0.0
        current_reward = 0
        self.env.force_fps = not_realtime
        self.env.reset_game()
        done = False
        while not done:
            state = self.get_state()
            action  = self._select_action(state)
            real_action = 119 if action == 1 else None
            action_string = 'eh, jump!' if action == 1 else 'erm, do nothing..'
            if debug and total_reward > current_reward:
                print(action_string, 'total rewards:', total_reward)
            current_reward = total_reward
            total_reward += self.env.act(real_action)
            done = self.env.game_over()
        print('game over!')
