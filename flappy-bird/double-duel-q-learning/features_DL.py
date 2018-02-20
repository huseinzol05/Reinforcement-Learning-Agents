import numpy as np
from ple import PLE
import random
from ple.games.flappybird import FlappyBird
import tensorflow as tf
from collections import deque
import os
import pickle

class Model:
    def __init__(self, input_size, output_size, layer_size, learning_rate):
        self.X = tf.placeholder(tf.float32, (None, input_size))
        self.Y = tf.placeholder(tf.float32, (None, output_size))
        input_layer = tf.Variable(tf.random_normal([self.INPUT_SIZE, self.LAYER_SIZE]))
        bias = tf.Variable(tf.random_normal([self.LAYER_SIZE]))
        action_layer = tf.Variable(tf.random_normal([self.LAYER_SIZE // 2, self.OUTPUT_SIZE]))
        validation_layer = tf.Variable(tf.random_normal([self.LAYER_SIZE // 2, 1]))
        feed_forward = tf.nn.relu(tf.matmul(self.X, input_layer) + bias)
        self.tensor_action, self.tensor_validation = tf.split(feed_forward,2,1)
        self.feed_action = tf.matmul(self.tensor_action, action_layer)
        self.feed_validation = tf.matmul(self.tensor_validation, action_layer)
        self.logits = self.feed_validation + tf.subtract(self.feed_action,tf.reduce_mean(self.feed_action,axis=1,keep_dims=True))
        self.cost = tf.reduce_sum(tf.square(self.Y - self.logits))
        self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(self.cost)

class Agent:

    LEARNING_RATE = 0.003
    BATCH_SIZE = 32
    INPUT_SIZE = 8
    LAYER_SIZE = 500
    OUTPUT_SIZE = 2
    EPSILON = 1
    DECAY_RATE = 0.005
    MIN_EPSILON = 0.1
    GAMMA = 0.99
    MEMORIES = deque()
    COPY = 1000
    T_COPY = 0
    MEMORY_SIZE = 300
    # based on documentation, features got 8 dimensions
    # output is 2 dimensions, 0 = do nothing, 1 = jump

    def __init__(self, screen=False, forcefps=True):
        self.game = FlappyBird(pipe_gap=125)
        self.env = PLE(self.game, fps=30, display_screen=screen, force_fps=forcefps)
        self.env.init()
        self.env.getGameState = self.game.getGameState
        self.model = Model(self.INPUT_SIZE, self.OUTPUT_SIZE, self.LAYER_SIZE, self.LEARNING_RATE)
        self.model_negative = Model(self.INPUT_SIZE, self.OUTPUT_SIZE, self.LAYER_SIZE, self.LEARNING_RATE)
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(tf.global_variables())
        self.rewards = []

    def _assign(self):
        for i in range(len(self.trainable)//2):
            assign_op = self.trainable[i+len(self.trainable)//2].assign(self.trainable[i])
            sess.run(assign_op)

    def _memorize(self, state, action, reward, new_state, dead):
        self.MEMORIES.append((state, action, reward, new_state, dead))
        if len(self.MEMORIES) > self.MEMORY_SIZE:
            self.MEMORIES.popleft()

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
        Q_new_negative = sess.run(self.model_negative.logits, feed_dict={self.model_negative.X:new_states})
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
            dead = False
            while not dead:
                if (self.T_COPY + 1) % self.COPY == 0:
                    self._assign()
                state = self.get_state()
                action  = self._select_action(state)
                real_action = 119 if action == 1 else None
                reward = self.env.act(real_action)
                total_reward += reward
                new_state = self.get_state()
                dead = self.env.game_over()
                self._memorize(state, action, reward, new_state, dead)
                batch_size = min(len(self.MEMORIES), self.BATCH_SIZE)
                replay = random.sample(self.MEMORIES, batch_size)
                X, Y = self._construct_memories(replay)
                cost, _ = self.sess.run([self.model.cost, self.model.optimizer], feed_dict={self.model.X: X, self.model.Y:Y})
                self.T_COPY += 1
            self.rewards.append(total_reward)
            self.EPSILON = self.MIN_EPSILON + (1.0 - self.MIN_EPSILON) * np.exp(-self.DECAY_RATE * i)
            if (i+1) % checkpoint == 0:
                print('epoch:', i + 1, 'total rewards:', total_reward)
                print('epoch:', i + 1, 'cost:', cost)

    def fit(self, iterations, checkpoint):
        self.get_reward(iterations, checkpoint)
