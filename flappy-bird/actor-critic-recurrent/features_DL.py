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
            self.X = tf.placeholder(tf.float32, (None, input_size))
            cell = tf.nn.rnn_cell.LSTMCell(512, state_is_tuple = False)
            self.hidden_layer = tf.placeholder(tf.float32, (None, 2 * 512))
            self.rnn,self.last_state = tf.nn.dynamic_rnn(inputs=self.X,cell=cell,
                                                        dtype=tf.float32,
                                                        initial_state=self.hidden_layer)
            w = tf.Variable(tf.random_normal([512, output_size]))
            self.logits = tf.matmul(self.rnn[:,-1], w)

class Critic:
    def __init__(self, name, input_size, output_size, size_layer, learning_rate):
        with tf.variable_scope(name):
            self.X = tf.placeholder(tf.float32, (None, input_size))
            self.Y = tf.placeholder(tf.float32, (None, output_size))
            self.REWARD = tf.placeholder(tf.float32, (None, 1))
            layer_merge = tf.Variable(tf.random_normal([output_size, size_layer//2]))
            layer_merge_out = tf.Variable(tf.random_normal([size_layer//2, 1]))
            cell = tf.nn.rnn_cell.LSTMCell(512, state_is_tuple = False)
            self.hidden_layer = tf.placeholder(tf.float32, (None, 2 * 512))
            self.rnn,self.last_state = tf.nn.dynamic_rnn(inputs=self.X,cell=cell,
                                                        dtype=tf.float32,
                                                        initial_state=self.hidden_layer)
            w = tf.Variable(tf.random_normal([512, output_size]))
            feed_critic = tf.nn.relu(tf.matmul(self.rnn[:,-1], w)) + self.Y
            feed_critic = tf.nn.relu(tf.matmul(feed_critic, layer_merge))
            self.logits = tf.matmul(feed_critic, layer_merge_out)
            self.cost = np.reduce_mean(tf.square(self.REWARD - self.logits))
            self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)

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
    INITIAL_FEATURES = np.zeros((4, INPUT_SIZE))
    MEMORIES = deque()
    MEMORY_SIZE = 300
    COPY = 1000
    T_COPY = 0
    # based on documentation, features got 8 dimensions
    # output is 2 dimensions, 0 = do nothing, 1 = jump

    def __init__(self, screen=False, forcefps=True):
        self.game = FlappyBird(pipe_gap=125)
        self.env = PLE(self.game, fps=30, display_screen=screen, force_fps=forcefps)
        self.env.init()
        self.env.getGameState = self.game.getGameState
        self.actor = Actor('actor', self.INPUT_SIZE, self.OUTPUT_SIZE, self.LAYER_SIZE)
        self.actor_target = Actor('actor-target', self.INPUT_SIZE, self.OUTPUT_SIZE, self.LAYER_SIZE)
        self.critic = Critic('critic', self.INPUT_SIZE, self.OUTPUT_SIZE, self.LAYER_SIZE, self.LEARNING_RATE)
        self.critic_target = Critic('critic-target', self.INPUT_SIZE, self.OUTPUT_SIZE, self.LAYER_SIZE, self.LEARNING_RATE)
        self.grad_critic = tf.gradients(self.critic.logits, self.critic.Y)
        self.actor_critic_grad = tf.placeholder(tf.float32, [None, self.OUTPUT_SIZE])
        weights_actor = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor')
        self.grad_actor = tf.gradients(self.actor.logits, weights_actor, -self.actor_critic_grad)
        grads = zip(self.grad_actor, weights_actor)
        self.optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE).apply_gradients(grads)
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

    def _construct_memories_and_train(self, replay):
        # state_r, action_r, reward_r, new_state_r, dead_r = replay
        # train actor
        states = np.array([a[0] for a in replay])
        new_states = np.array([a[3] for a in replay])
        init_values = np.array([a[-1] for a in replay])
        Q = self.sess.run(self.actor.logits, feed_dict={self.actor.X: states, self.actor.hidden_layer:init_values})
        Q_target = self.sess.run(self.actor_target.logits, feed_dict={self.actor_target.X: states, self.actor_target.hidden_layer:init_values})
        grads = self.sess.run(self.grad_critic, feed_dict={self.critic.X:states, self.critic.hidden_layer: init_values, self.critic.Y:Q})
        self.sess.run(self.optimizer, feed_dict={self.actor.X:states, self.actor.hidden_layer:init_values, self.actor_critic_grad:grads})

        # train critic
        rewards = np.array([a[2] for a in replay]).reshape((-1, 1))
        rewards_target = self.sess.run(self.critic_target.logits, feed_dict={self.critic_target.X:new_states, self.critic_target.hidden_layer: init_values,
                                                                             self.critic_target.Y:Q_target})
        for i in range(len(replay)):
            if not replay[0][-1]:
                rewards[i,0] += self.GAMMA * rewards_target
        cost, _ = self.sess.run([self.critic.cost, self.critic.optimizer), feed_dict={self.critic.X:states, self.critic.hidden_layer: init_values,
                                                                                      self.critic.Y:Q, self.critic.REWARD:rewards})
        return cost

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
                    self._assign('actor', 'actor-target')
                    self._assign('critic', 'critic-target')
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
                cost = self._construct_memories_and_train(replay)
                self.EPSILON = self.MIN_EPSILON + (1.0 - self.MIN_EPSILON) * np.exp(-self.DECAY_RATE * i)
                self.T_COPY += 1
            self.rewards.append(total_reward)
            if (i+1) % checkpoint == 0:
                print('epoch:', i + 1, 'total rewards:', total_reward)
                print('epoch:', i + 1, 'cost:', cost)

    def fit(self, iterations, checkpoint):
        self.get_reward(iterations, checkpoint)
