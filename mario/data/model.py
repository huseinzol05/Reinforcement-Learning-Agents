import tensorflow as tf
import numpy as np
from collections import deque
from scipy.misc import imresize

class Model:

    LEARNING_RATE = 1e-6
    BATCH_SIZE = 32
    EPSILON = 1
    DECAY_RATE = 0.005
    MIN_EPSILON = 0.1
    GAMMA = 0.99
    MEMORIES = deque()
    INITIAL_GAME = True
    MEMORY_SIZE = 50000
    INITIAL_IMAGES = np.zeros((80, 80, 4))
    # left, right, right + jump, left + jump, jump, do nothing
    OUTPUT_SIZE = 6

    def __init__(self):
        def conv_layer(x, conv, stride = 1):
            return tf.nn.conv2d(x, conv, [1, stride, stride, 1], padding = 'SAME')

        def pooling(x, k = 2, stride = 2):
            return tf.nn.max_pool(x, ksize = [1, k, k, 1], strides = [1, stride, stride, 1], padding = 'SAME')

        self.X = tf.placeholder(tf.float32, [None, 80, 80, 4])
        self.Y = tf.placeholder(tf.float32, [None, self.OUTPUT_SIZE])
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
        w_fc1 = tf.Variable(tf.truncated_normal([pulling_size, 512], stddev = 0.1))
        b_fc1 = tf.Variable(tf.truncated_normal([512], stddev = 0.01))
        w_fc2 = tf.Variable(tf.truncated_normal([512, self.OUTPUT_SIZE], stddev = 0.1))
        b_fc2 = tf.Variable(tf.truncated_normal([self.OUTPUT_SIZE], stddev = 0.01))
        fc_1 = tf.nn.relu(tf.matmul(conv3, w_fc1) + b_fc1)
        self.logits = tf.matmul(fc_1, w_fc2)  + b_fc2
        self.cost = tf.reduce_sum(tf.square(self.Y - self.logits))
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.LEARNING_RATE).minimize(self.cost)
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(tf.global_variables())

    def memorize(self, state, action, reward, new_state, dead):
        self.MEMORIES.append((state, action, reward, new_state, dead))
        if len(self.MEMORIES) > self.MEMORY_SIZE:
            self.MEMORIES.popleft()

    def get_image(self, image):
        r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return imresize(gray, size = (80, 80))

    def get_predicted_action(self, sequence):
        prediction = self.predict(np.array(sequence))[0]
        return np.argmax(prediction)

    def select_action(self, state):
        if np.random.rand() < self.EPSILON:
            action = np.random.randint(self.OUTPUT_SIZE)
        else:
            action = self.get_predicted_action([state])
        self.EPSILON -= (self.EPSILON / self.MEMORY_SIZE)
        return action

    def predict(self, inputs):
        return self.sess.run(self.logits, feed_dict={self.X:inputs})

    def construct_memories(self, replay):
        states = np.array([a[0] for a in replay])
        new_states = np.array([a[3] for a in replay])
        Q = self.predict(states)
        Q_new = self.predict(new_states)
        replay_size = len(replay)
        X = np.empty((replay_size, 80, 80, 4))
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
