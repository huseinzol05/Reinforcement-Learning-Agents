import tensorflow as tf
import numpy as np
from collections import deque
from setting import *

class Model:
	def __init__(self):
		self.on_drive = False
		self.memory = deque()
		self.initial_sensor = np.zeros((INITIAL_MEMORY, ACTIONS))
		self.X = tf.placeholder("float", [None, ACTIONS])
		self.Y = tf.placeholder("float", [None])
		self.actions = []
		for i in range(ACTIONS):
			self.actions.append(tf.placeholder("float", [None, 1]))
			
		w_1 = tf.Variable(tf.truncated_normal([ACTIONS, 128], stddev = 0.1))
		b_1 = tf.Variable(tf.truncated_normal([128], stddev = 0.01))
		w_2 = tf.Variable(tf.truncated_normal([128, 128], stddev = 0.1))
		b_2 = tf.Variable(tf.truncated_normal([128], stddev = 0.01))
		
		self.w_actions, self.b_actions = [], []
		for i in range(ACTIONS):
			self.w_actions.append(tf.Variable(tf.truncated_normal([128, 1], stddev = 0.1)))
			self.b_actions.append(tf.Variable(tf.truncated_normal([1], stddev = 0.01)))
			
		feed_forward = tf.nn.sigmoid(tf.matmul(self.X, w_1) + b_1)
		feed_forward = tf.nn.sigmoid(tf.matmul(feed_forward, w_2) + b_2)
		
		self.logits = []
		for i in range(ACTIONS):
			self.logits.append(tf.matmul(feed_forward, self.w_actions[i]) + self.b_actions[i])
			
		self.readout = []
		for i in range(ACTIONS):
			self.readout.append(tf.reduce_sum(tf.multiply(self.logits[i], self.actions[i]), reduction_indices = 1))
			
		self.cost = sum([tf.sqrt(tf.reduce_mean(tf.square(self.Y - readout))) for readout in self.readout])
		self.optimizer = tf.train.AdamOptimizer(0.01).minimize(self.cost)
		