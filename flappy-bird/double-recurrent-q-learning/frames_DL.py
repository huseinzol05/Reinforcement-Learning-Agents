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
    def __init__(self, output_size, learning_rate, batch_size, name):
        def conv_layer(x, conv, stride = 1):
            return tf.nn.conv2d(x, conv, [1, stride, stride, 1], padding = 'SAME')
        def pooling(x, k = 2, stride = 2):
            return tf.nn.max_pool(x, ksize = [1, k, k, 1], strides = [1, stride, stride, 1], padding = 'SAME')
        with tf.variable_scope(name):
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
            conv3 = tf.reshape(tf.reshape(conv3, [-1, pulling_size]), [batch_size, 8, 512])
            cell = tf.nn.rnn_cell.LSTMCell(512, state_is_tuple = False)
            self.hidden_layer = tf.placeholder(tf.float32, (None, 2 * 512))
            self.rnn,self.last_state = tf.nn.dynamic_rnn(inputs=conv3,cell=cell,
                                                        dtype=tf.float32,
                                                        initial_state=self.hidden_layer)
            w = tf.Variable(tf.random_normal([512, output_size]))
            self.logits = tf.matmul(self.rnn[:,-1], w)
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
    MEMORY_SIZE = 1000
    COPY = 1000
    T_COPY = 0
    INITIAL_IMAGES = np.zeros((80, 80, 4))
    # output is 2 dimensions, 0 = do nothing, 1 = jump

    def __init__(self, screen=False, forcefps=True):
        self.game = FlappyBird(pipe_gap=125)
        self.env = PLE(self.game, fps=30, display_screen=screen, force_fps=forcefps)
        self.env.init()
        self.env.getGameState = self.game.getGameState
        # output_size, learning_rate, batch_size, name
        self.model = Model(self.OUTPUT_SIZE, self.LEARNING_RATE, self.BATCH_SIZE, 'real_model')
        self.model_negative = Model(self.OUTPUT_SIZE, self.LEARNING_RATE, self.BATCH_SIZE, 'negative_model')
        self.sess = tf.InteractiveSession()
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(tf.global_variables())
        self.rewards = []

    def _assign(self, from_name, to_name):
        from_w = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=from_name)
        to_w = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=to_name)
        for i in range(len(from_w)):
            assign_op = to_w[i].assign(from_w[i])
            sess.run(assign_op)

    def _memorize(self, state, action, reward, new_state, dead, rnn_state):
        self.MEMORIES.append((state, action, reward, new_state, dead, rnn_state))
        if len(self.MEMORIES) > self.MEMORY_SIZE:
            self.MEMORIES.popleft()

    def _get_image(self, image):
        r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return imresize(gray, size = (80, 80))

    def _construct_memories(self, replay):
        states = np.array([a[0] for a in replay])
        new_states = np.array([a[3] for a in replay])
        init_values = np.array([a[-1] for a in replay])
        Q = sess.run(self.model.logits, feed_dict={self.model.X:states, self.model.hidden_layer:init_values})
        Q_new = sess.run(self.model.logits, feed_dict={self.model.X:new_states, self.model.hidden_layer:init_values})
        Q_new_negative = sess.run(self.model_negative.logits, feed_dict={self.model_negative.X:new_states, self.model_negative.hidden_layer:init_values})
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
        return X, Y, init_values

    def save(self, checkpoint_name):
        self.saver.save(self.sess, os.getcwd() + "/%s.ckpt" %(checkpoint_name))
        with open('%s-acc.p'%(checkpoint_name), 'wb') as fopen:
            pickle.dump(self.rewards, fopen)

    def load(self, checkpoint_name):
        self.saver.restore(self.sess, os.getcwd() + "/%s.ckpt" %(checkpoint_name))
        with open('%s-acc.p'%(checkpoint_name), 'rb') as fopen:
            self.rewards = pickle.load(fopen)

    def get_reward(self, iterations, checkpoint):
        for i in range(iterations):
            total_reward = 0
            self.env.reset_game()
            state = self._get_image(self.env.getScreenRGB())
            for k in range(self.INITIAL_IMAGES.shape[2]):
                self.INITIAL_IMAGES[:,:,k] = state
            dead = False
            init_value = np.zeros((1, 2 * 512))
            while not dead:
                if (self.T_COPY + 1) % self.COPY == 0:
                    self._assign('real_model', 'target_model')
                if np.random.rand() < self.EPSILON:
                    action = np.random.randint(self.OUTPUT_SIZE)
                else:
                    action, last_state = sess.run(self.model.logits,
                                                  self.model.last_state,
                                                  feed_dict={self.model.X:[self.INITIAL_IMAGES],
                                                             self.model.hidden_layer:init_values})
                    action, init_value = np.argmax(action[0]), last_state[0]
                real_action = 119 if action == 1 else None
                reward = self.env.act(real_action)
                total_reward += reward
                state = self._get_image(self.env.getScreenRGB())
                new_state = np.append(state.reshape([80, 80, 1]), self.INITIAL_IMAGES[:, :, :3], axis = 2)
                dead = self.env.game_over()
                self._memorize(self.INITIAL_IMAGES, action, reward, new_state, dead, init_value)
                batch_size = min(len(self.MEMORIES), self.BATCH_SIZE)
                replay = random.sample(self.MEMORIES, batch_size)
                X, Y, init_values = self._construct_memories(replay)
                cost, _ = self.sess.run([self.model.cost, self.model.optimizer], feed_dict={self.model.X: X, self.model.Y:Y,
                                                                                self.model.hidden_layer: init_values})
                self.INITIAL_IMAGES = new_state
                self.T_COPY += 1
            self.rewards.append(total_reward)
            self.EPSILON = self.MIN_EPSILON + (1.0 - self.MIN_EPSILON) * np.exp(-self.DECAY_RATE * i)
            if (i+1) % checkpoint == 0:
                print('epoch:', i + 1, 'total rewards:', total_reward)
                print('epoch:', i + 1, 'cost:', cost)

    def fit(self, iterations, checkpoint):
        self.get_reward(iterations, checkpoint)
