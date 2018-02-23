import numpy as np
from ple import PLE
import random
from ple.games.flappybird import FlappyBird
import tensorflow as tf
from collections import deque
import os
import pickle

class Model:
    def __init__(self, input_size, output_size, layer_size, learning_rate, name):
        with tf.variable_scope(name):
            self.X = tf.placeholder(tf.float32, (None, None, input_size))
            self.Y = tf.placeholder(tf.float32, (None, output_size))
            cell = tf.nn.rnn_cell.LSTMCell(512, state_is_tuple = False)
            self.hidden_layer = tf.placeholder(tf.float32, (None, 2 * 512))
            self.rnn,self.last_state = tf.nn.dynamic_rnn(inputs=self.X,cell=cell,
                                                        dtype=tf.float32,
                                                        initial_state=self.hidden_layer)
            self.tensor_action, self.tensor_validation = tf.split(self.rnn[:, -1,:],2,1)
            self.feed_action = tf.matmul(self.tensor_action, action_layer)
            self.feed_validation = tf.matmul(self.tensor_validation, action_layer)
            self.logits = self.feed_validation + tf.subtract(self.feed_action,tf.reduce_mean(self.feed_action,axis=1,keep_dims=True))
            self.cost = tf.reduce_sum(tf.square(self.Y - self.logits))
            self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(self.cost)

class Agent:

    LEARNING_RATE = 0.003
    BATCH_SIZE = 32
    EPSILON = 1
    DECAY_RATE = 0.005
    MIN_EPSILON = 0.1
    GAMMA = 0.99
    MEMORIES = deque()
    COPY = 1000
    T_COPY = 0
    MEMORY_SIZE = 300
    INITIAL_FEATURES = np.zeros((4, INPUT_SIZE))
    INPUT_SIZE = 8
    LAYER_SIZE = 500
    OUTPUT_SIZE = 2
    # based on documentation, features got 8 dimensions
    # output is 2 dimensions, 0 = do nothing, 1 = jump

    def __init__(self, screen=False, forcefps=True):
        self.game = FlappyBird(pipe_gap=125)
        self.env = PLE(self.game, fps=30, display_screen=screen, force_fps=forcefps)
        self.env.init()
        self.env.getGameState = self.game.getGameState
        # input_size, output_size, layer_size, learning_rate, name
        self.model = Model(self.INPUT_SIZE, self.OUTPUT_SIZE, self.LAYER_SIZE, self.LEARNING_RATE, 'real_model')
        self.model_negative = Model(self.INPUT_SIZE, self.OUTPUT_SIZE, self.LAYER_SIZE, self.LEARNING_RATE, 'negative_model')
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

    def _construct_memories(self, replay):
        states = np.array([a[0] for a in replay])
        new_states = np.array([a[3] for a in replay])
        init_values = np.array([a[-1] for a in replay])
        Q = sess.run(self.model.logits, feed_dict={self.model.X:states, self.model_negative.hidden_layer:init_values})
        Q_new = sess.run(self.model.logits, feed_dict={self.model.X:new_states, self.model.hidden_layer:init_values})
        Q_new_negative = sess.run(self.model_negative.logits, feed_dict={self.model_negative.X:new_states, self.model_negative.hidden_layer:init_values})
        replay_size = len(replay)
        X = np.empty((replay_size, self.INPUT_SIZE))
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

    def save(self, checkpoint_name):
        self.saver.save(self.sess, os.getcwd() + "/%s.ckpt" %(checkpoint_name))
        with open('%s-acc.p'%(checkpoint_name), 'wb') as fopen:
            pickle.dump(self.rewards, fopen)

    def load(self, checkpoint_name):
        self.saver.restore(self.sess, os.getcwd() + "/%s.ckpt" %(checkpoint_name))
        with open('%s-acc.p'%(checkpoint_name), 'rb') as fopen:
            self.rewards = pickle.load(fopen)

    def get_state(self):
        state = self.env.getGameState()
        return np.array(list(state.values()))

    def get_reward(self, iterations, checkpoint):
        for i in range(iterations):
            total_reward = 0
            self.env.reset_game()
            dead = False
            init_value = np.zeros((1, 2 * 512))
            state = self.get_state()
            for i in range(self.INITIAL_FEATURES.shape[0]):
                self.INITIAL_FEATURES[i,:] = state
            while not dead:
                if (self.T_COPY + 1) % self.COPY == 0:
                    self._assign('real_model', 'target_model')
                if np.random.rand() < self.EPSILON:
                    action = np.random.randint(self.OUTPUT_SIZE)
                else:
                    action, last_state = sess.run(self.model.logits,
                                                  self.model.last_state,
                                                  feed_dict={self.model.X:[self.INITIAL_FEATURES],
                                                             self.model.hidden_layer:init_values})
                    action, init_value = np.argmax(action[0]), last_state[0]
                real_action = 119 if action == 1 else None
                reward = self.env.act(real_action)
                total_reward += reward
                new_state = np.append(self.get_state(), self.INITIAL_FEATURES[:3, :], axis = 0)
                dead = self.env.game_over()
                self._memorize(state, action, reward, new_state, dead, init_value)
                batch_size = min(len(self.MEMORIES), self.BATCH_SIZE)
                replay = random.sample(self.MEMORIES, batch_size)
                X, Y, init_values = self._construct_memories(replay)
                cost, _ = self.sess.run([self.model.cost, self.model.optimizer], feed_dict={self.model.X: X, self.model.Y:Y,
                                                                                self.model.hidden_layer: init_values})
                self.T_COPY += 1
            self.rewards.append(total_reward)
            self.EPSILON = self.MIN_EPSILON + (1.0 - self.MIN_EPSILON) * np.exp(-self.DECAY_RATE * i)
            if (i+1) % checkpoint == 0:
                print('epoch:', i + 1, 'total rewards:', total_reward)
                print('epoch:', i + 1, 'cost:', cost)

    def fit(self, iterations, checkpoint):
        self.get_reward(iterations, checkpoint)
