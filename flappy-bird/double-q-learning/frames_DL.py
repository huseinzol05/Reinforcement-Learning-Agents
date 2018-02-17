import numpy as np
from ple import PLE
import random
from ple.games.flappybird import FlappyBird
import tensorflow as tf
from collections import deque
from scipy.misc import imresize
import os
import pickle

class Model:
    def __init__(self, output_size, learning_rate):
        def conv_layer(x, conv, stride = 1):
            return tf.nn.conv2d(x, conv, [1, stride, stride, 1], padding = 'SAME')
        def pooling(x, k = 2, stride = 2):
            return tf.nn.max_pool(x, ksize = [1, k, k, 1], strides = [1, stride, stride, 1], padding = 'SAME')
        self.X = tf.placeholder(tf.float32, [None, 80, 80, 4])
        self.Y = tf.placeholder(tf.float32, [None, output_size])
        self.w_conv1 = tf.Variable(tf.truncated_normal([8, 8, 4, 32], stddev = 0.1))
        self.b_conv1 = tf.Variable(tf.truncated_normal([32], stddev = 0.01))
        conv1 = tf.nn.relu(conv_layer(self.X, self.w_conv1, stride = 4) + self.b_conv1)
        pooling1 = pooling(conv1)
        self.w_conv2 = tf.Variable(tf.truncated_normal([4, 4, 32, 64], stddev = 0.1))
        self.b_conv2 = tf.Variable(tf.truncated_normal([64], stddev = 0.01))
        conv2 = tf.nn.relu(conv_layer(pooling1, self.w_conv2, stride = 2) + self.b_conv2)
        self.w_conv3 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev = 0.1))
        self.b_conv3 = tf.Variable(tf.truncated_normal([64], stddev = 0.01))
        conv3 = tf.nn.relu(conv_layer(conv2, self.w_conv3) + self.b_conv3)
        pulling_size = int(conv3.shape[1]) * int(conv3.shape[2]) * int(conv3.shape[3])
        conv3 = tf.reshape(conv3, [-1, pulling_size])
        self.w_fc1 = tf.Variable(tf.truncated_normal([pulling_size, 256], stddev = 0.1))
        self.b_fc1 = tf.Variable(tf.truncated_normal([256], stddev = 0.01))
        self.w_fc2 = tf.Variable(tf.truncated_normal([256, 2], stddev = 0.1))
        self.b_fc2 = tf.Variable(tf.truncated_normal([2], stddev = 0.01))
        fc_1 = tf.nn.relu(tf.matmul(conv3, self.w_fc1) + self.b_fc1)
        self.logits = tf.matmul(fc_1, self.w_fc2)  + self.b_fc2
        self.cost = tf.reduce_sum(tf.square(self.Y - self.logits))
        self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(self.cost)

class Agent:

    LEARNING_RATE = 1e-6
    BATCH_SIZE = 32
    OUTPUT_SIZE = 2
    EPSILON = 1
    DECAY_RATE = 0.005
    MIN_EPSILON = 0.1
    GAMMA = 0.99
    MEMORIES = deque()
    MEMORY_SIZE = 300
    COPY = 1000
    T_COPY = 0
    INITIAL_IMAGES = np.zeros((80, 80, 4))
    # based on documentation, features got 8 dimensions
    # output is 2 dimensions, 0 = do nothing, 1 = jump

    def __init__(self, screen=False, forcefps=True):
        self.game = FlappyBird(pipe_gap=125)
        self.env = PLE(self.game, fps=30, display_screen=screen, force_fps=forcefps)
        self.env.init()
        self.env.getGameState = self.game.getGameState
        self.model = Model(self.OUTPUT_SIZE, self.LEARNING_RATE)
        self.model_negative = Model(self.OUTPUT_SIZE, self.LEARNING_RATE)
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(tf.global_variables())
        self.trainable = tf.trainable_variables()
        self.rewards = []

    def _assign(self):
        for i in range(len(self.trainable)//2):
            assign_op = self.trainable[i+len(self.trainable)//2].assign(self.trainable[i])
            sess.run(assign_op)

    def _memorize(self, state, action, reward, new_state, dead):
        self.MEMORIES.append((state, action, reward, new_state, dead))
        if len(self.MEMORIES) > self.MEMORY_SIZE:
            self.MEMORIES.popleft()

    def _get_image(self, image):
        r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return imresize(gray, size = (80, 80))

    def _select_action(self, state):
        if np.random.rand() < self.EPSILON:
            action = np.random.randint(self.OUTPUT_SIZE)
        else:
            action = self.get_predicted_action([state])
        return action

    def _construct_memories(self, replay):
        states = np.array([a[0] for a in replay])
        new_states = np.array([a[3] for a in replay])
        Q = self.predict(states)
        Q_new = self.predict(new_states)
        Q_new_negative = sess.run(self.model_negative, feed_dict={self.model_negative.X:new_states})
        replay_size = len(replay)
        X = np.empty((replay_size, 80, 80, 4))
        Y = np.empty((replay_size, self.OUTPUT_SIZE))
        for i in range(replay_size):
            state_r, action_r, reward_r, new_state_r, dead_r = replay[i]
            target = Q[i]
            target[action_r] = reward_r
            if not dead_r:
                target[action_r] += self.GAMMA * Q_new_negative[i, np.argmax(Q_new[i])]
            X[i] = state_r
            Y[i] = target
        return X, Y

    def predict(self, inputs):
        return self.sess.run(self.model.logits, feed_dict={self.model.X:inputs})

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
            total_reward = 0
            self.env.reset_game()
            state = self._get_image(self.env.getScreenRGB())
            for k in range(self.INITIAL_IMAGES.shape[2]):
                self.INITIAL_IMAGES[:,:,k] = state
            dead = False
            while not dead:
                if (self.T_COPY + 1) % self.COPY == 0:
                    self._assign()
                action  = self._select_action(self.INITIAL_IMAGES)
                real_action = 119 if action == 1 else None
                reward = self.env.act(real_action)
                total_reward += reward
                new_state = self.get_state()
                state = self._get_image(self.env.getScreenRGB())
                new_state = np.append(state.reshape([80, 80, 1]), self.INITIAL_IMAGES[:, :, :3], axis = 2)
                dead = self.env.game_over()
                self._memorize(self.INITIAL_IMAGES, action, reward, new_state, dead)
                batch_size = min(len(self.MEMORIES), self.BATCH_SIZE)
                replay = random.sample(self.MEMORIES, batch_size)
                X, Y = self._construct_memories(replay)
                cost, _ = self.sess.run([self.cost, self.optimizer], feed_dict={self.X: X, self.Y:Y})
                self.INITIAL_IMAGES = new_state
                self.T_COPY += 1
            self.rewards.append(total_reward)
            self.EPSILON = self.MIN_EPSILON + (1.0 - self.MIN_EPSILON) * np.exp(-self.DECAY_RATE * i)
            if (i+1) % checkpoint == 0:
                print('epoch:', i + 1, 'total rewards:', total_reward)
                print('epoch:', i + 1, 'cost:', cost)

    def fit(self, iterations, checkpoint):
        self.get_reward(iterations, checkpoint)
