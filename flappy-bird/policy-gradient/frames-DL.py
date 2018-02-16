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
    INITIAL_IMAGES = np.zeros((80, 80, 4))
    INPUT_SIZE = 8
    # based on documentation, features got 8 dimensions
    OUTPUT_SIZE = 2
    # output is 2 dimensions, 0 = do nothing, 1 = jump

    def __init__(self, screen=False, forcefps=True):
        self.game = FlappyBird(pipe_gap=125)
        self.env = PLE(self.game, fps=30, display_screen=screen, force_fps=forcefps)
        self.env.init()
        self.env.getGameState = self.game.getGameState
        def conv_layer(x, conv, stride = 1):
            return tf.nn.conv2d(x, conv, [1, stride, stride, 1], padding = 'SAME')
        def pooling(x, k = 2, stride = 2):
            return tf.nn.max_pool(x, ksize = [1, k, k, 1], strides = [1, stride, stride, 1], padding = 'SAME')
        self.X = tf.placeholder(tf.float32, [None, 80, 80, 4])
        self.REWARDS = tf.placeholder(tf.float32, (None))
        self.ACTIONS = tf.placeholder(tf.int32, (None))
        w_conv1 = tf.Variable(tf.truncated_normal([8, 8, 4, 32], stddev = 0.1))
        b_conv1 = tf.Variable(tf.truncated_normal([32], stddev = 0.01))
        conv1 = tf.nn.relu(conv_layer(self.X, w_conv1, stride = 4) + b_conv1)
        pooling1 = pooling(conv1)
        w_conv2 = tf.Variable(tf.truncated_normal([4, 4, 32, 64], stddev = 0.1))
        b_conv2 = tf.Variable(tf.truncated_normal([64], stddev = 0.01))
        conv2 = tf.nn.relu(conv_layer(pooling1, w_conv2, stride = 2) + b_conv2)
        w_conv3 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev = 0.1))
        b_conv3 = tf.Variable(tf.truncated_normal([64], stddev = 0.01))
        conv3 = tf.nn.relu(conv_layer(conv2, w_conv3) + b_conv3)
        pulling_size = int(conv3.shape[1]) * int(conv3.shape[2]) * int(conv3.shape[3])
        conv3 = tf.reshape(conv3, [-1, pulling_size])
        w_fc1 = tf.Variable(tf.truncated_normal([pulling_size, 256], stddev = 0.1))
        b_fc1 = tf.Variable(tf.truncated_normal([256], stddev = 0.01))
        w_fc2 = tf.Variable(tf.truncated_normal([256, 2], stddev = 0.1))
        b_fc2 = tf.Variable(tf.truncated_normal([2], stddev = 0.01))
        fc_1 = tf.nn.relu(tf.matmul(conv3, w_fc1) + b_fc1)
        self.logits = tf.nn.softmax(tf.matmul(fc_1, w_fc2)  + b_fc2)
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

    def _get_image(self, image):
        r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return imresize(gray, size = (80, 80))

    def get_reward(self, iterations, checkpoint):
        for i in range(iterations):
            ep_history = []
            for k in range(self.EPISODE):
                total_reward = 0
                self.env.reset_game()
                done = False
                state = self._get_image(self.env.getScreenRGB())
                for i in range(self.INITIAL_IMAGES.shape[2]):
                    self.INITIAL_IMAGES[:,:,i] = state
                sequence = [self.INITIAL_IMAGES]
                while not done:
                    action = self._select_action(state)
                    real_action = 119 if action == 1 else None
                    reward = self.env.act(real_action)
                    reward += random.choice([0.0001, -0.0001])
                    total_reward += reward
                    next_state = np.append(self._get_image(self.env.getScreenRGB()).reshape([80, 80, 1]), self.INITIAL_IMAGES[:, :, :3], axis = 2)
                    ep_history.append([self.INITIAL_IMAGES,action,total_reward,next_state])
                    self.INITIAL_IMAGES = next_state
                    sequence = [self.INITIAL_IMAGES]
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
