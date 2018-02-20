import numpy as np
from ple import PLE
import random
from ple.games.flappybird import FlappyBird
import tensorflow as tf
from collections import deque
import os
import pickle

class Actor:
    def __init__(self, name, input_size, output_size, size_layer):
        with tf.variable_scope(name):
            self.X_actor = tf.placeholder(tf.float32, (None, input_size))
            layer_actor = tf.Variable(tf.random_normal([input_size, size_layer]))
            output_actor = tf.Variable(tf.random_normal([size_layer, output_size]))
            feed_actor = tf.nn.relu(tf.matmul(self.X_actor, layer_actor))
            self.logits_actor = tf.matmul(feed_actor, output_actor)

class Critic:
    def __init__(self, name, input_size, output_size, size_layer):
        with tf.variable_scope(name):
            self.X_critic = tf.placeholder(tf.float32, (None, input_size))
            self.Y_critic = tf.placeholder(tf.float32, (None, output_size))
            layer_critic = tf.Variable(tf.random_normal([input_size, size_layer]))
            output_critic = tf.Variable(tf.random_normal([size_layer, output_size]))
            feed_actor = tf.nn.relu(tf.matmul(self.X_actor, layer_actor))
            self.logits_critic = tf.matmul(feed_actor, output_actor) + self.Y_critic

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
    MEMORY_SIZE = 300
    # based on documentation, features got 8 dimensions
    # output is 2 dimensions, 0 = do nothing, 1 = jump

    def __init__(self, screen=False, forcefps=True):
        self.game = FlappyBird(pipe_gap=125)
        self.env = PLE(self.game, fps=30, display_screen=screen, force_fps=forcefps)
        self.env.init()
        self.env.getGameState = self.game.getGameState
        self.actor = Actor('actor', self.INPUT_SIZE, self.OUTPUT_SIZE, self.LAYER_SIZE)
        self.actor_target = Actor('actor-target', self.INPUT_SIZE, self.OUTPUT_SIZE, self.LAYER_SIZE)
        self.critic = Critic('critic', self.INPUT_SIZE, self.OUTPUT_SIZE, self.LAYER_SIZE)
        self.critic_target = Critic('critic-target', self.INPUT_SIZE, self.OUTPUT_SIZE, self.LAYER_SIZE)
        self.grad_critic = tf.gradients(self.critic.logits_critic, self.critic.Y_critic)
        self.actor_critic_grad = tf.placeholder(tf.float32, [None, self.OUTPUT_SIZE])
        weights_actor = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor')
        self.grad_actor = tf.gradients(self.model.logits_actor, weights_actor, -self.actor_critic_grad)
        grads = zip(self.grad_actor, weights_actor)
        self.optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE).apply_gradients(grads)
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(tf.global_variables())
        self.rewards = []

    def _assign(self, from_name, to_name):
        from_w = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=from_name)
        to_w = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=to_name)
        for i in range(len(from_w)):
            assign_op = to_w[i].assign(from_w[i])
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
        replay_size = len(replay)
        X = np.empty((replay_size, self.INPUT_SIZE))
        Y = np.empty((replay_size, self.OUTPUT_SIZE))
        for i in range(replay_size):
            state_r, action_r, reward_r, new_state_r, dead_r = replay[i]
            target = Q[i]
            target[action_r] = reward_r
            if not dead_r:
                target[action_r] += self.GAMMA * np.amax(Q_new[i])
            X[i] = state_r
            Y[i] = target
        return X, Y

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
            total_reward = 0
            self.env.reset_game()
            dead = False
            while not dead:
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
                cost, _ = self.sess.run([self.cost, self.optimizer], feed_dict={self.X: X, self.Y:Y})
            self.rewards.append(total_reward)
            self.EPSILON = self.MIN_EPSILON + (1.0 - self.MIN_EPSILON) * np.exp(-self.DECAY_RATE * i)
            if (i+1) % checkpoint == 0:
                print('epoch:', i + 1, 'total rewards:', total_reward)
                print('epoch:', i + 1, 'cost:', cost)

    def fit(self, iterations, checkpoint):
        self.get_reward(iterations, checkpoint)
