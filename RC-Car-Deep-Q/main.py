from socketIO_client import SocketIO, BaseNamespace
import tensorflow as tf
import numpy as np
import model_digital
import model_analog
import time
import json
import os
import realtime
import random
from setting import *
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
sns.set()

epsilon = INITIAL_EPSILON
time_elapsed, loss = 0, 0
sess = tf.InteractiveSession()
if USE_ANALOG:
	model_nn = model_analog.Model()
else:
	model_nn = model_digital.Model()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(tf.global_variables())
fig, axes = plt.subplots(figsize = (3, 3))
display = realtime.RealtimePlot(axes)

# action = [0, 0, 0, 0] = left, right, forward, backward
def on_car_response(*args):
	global car_namespace
	global time_elapsed
	global loss
	global epsilon
	sensors = json.loads(args[0])
	print sensors
	sensors = np.array(sensors)
	sensors_log = np.log1p(sensors)
	sensors_state = sensors * 0.034 / 2;
	sensors_mean = np.mean(sensors_state)
	hitted = True if np.where(sensors_state <= 1)[0].shape[0] > 0 else False
	if not model_nn.on_drive:
		for i in range(INITIAL_MEMORY):
			model_nn.initial_sensor[i, :] = sensors_log
		model_nn.on_drive = True
	action = np.zeros([ACTIONS], dtype = np.int)
	if time_elapsed % FRAME_PER_ACTION == 0:
		if random.random() <= epsilon:
			print('step: ', time_elapsed, ', do random actions')
			if not USE_ANALOG: 
				action = np.random.randint(2, size = ACTIONS)
			else:
				action = np.random.random(size = ACTIONS)
		else:
			if not USE_ANALOG:
				actions = sess.run([i for i in model_nn.logits], feed_dict = {model_nn.X: np.mean(model_nn.initial_sensor, axis = 0).reshape((-1, ACTIONS))})
				actions = np.array(actions).reshape((-1, 2))
				action[np.where(np.argmax(actions, axis = 1) == 1)[0]] = 1
			else:
				actions = sess.run([tf.nn.sigmoid(i) for i in model_nn.logits], feed_dict = {model_nn.X: np.mean(model_nn.initial_sensor, axis = 0).reshape((-1, ACTIONS))})
				actions = np.array(actions).reshape((-1))
				action = actions
		if np.argmax(action[:2]) == 0:
			action[1] = 0
		else:
			action[0] = 0
		if np.argmax(action[2:]) == 0:
			action[3] = 0
		else:
			action[2] = 0
	print('our action: ', action)
	car_namespace.emit('carupdate', json.dumps(action.tolist()))
	if epsilon > FINAL_EPSILON and time_elapsed > OBSERVE:
		epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
	stack_sensors = np.append(sensors_log.reshape([-1, ACTIONS]), model_nn.initial_sensor[:3, :], axis = 0)
	model_nn.memory.append((np.mean(model_nn.initial_sensor, axis = 0), action, sensors_mean, np.mean(stack_sensors, axis = 0), hitted))
	if len(model_nn.memory) > REPLAY_MEMORY_SIZE:
		model_nn.memory.popleft()
	if time_elapsed > OBSERVE:
		minibatch = random.sample(model_nn.memory, BATCH)
		initial_sensor_batch = [d[0] for d in minibatch]
		action_batch = [d[1] for d in minibatch]
		reward_batch = [d[2] for d in minibatch]
		sensor_batch = [d[3] for d in minibatch]
		y_batch = []
		if not USE_ANALOG: 
			actions_map = np.zeros((ACTIONS, BATCH, 2))
		else:
			actions_map = np.zeros((ACTIONS, BATCH, 1))
		actions = sess.run([i for i in model_nn.logits], feed_dict = {model_nn.X: sensor_batch})
		for i in range(len(minibatch)):
			if minibatch[i][4]:
				if not USE_ANALOG:
					y_batch.append(reward_batch[i])
				else:
					y_batch.append(np.log1p(reward_batch[i]))
			else:
				if not USE_ANALOG:
					y_batch.append(reward_batch[i] + np.sum(np.array([GAMMA * np.argmax(act[i]) for act in actions]) * [1, 1, 0.2, 1]))
				else:
					y_batch.append(np.log1p(reward_batch[i] + np.sum(np.array([GAMMA * act[i][0] for act in actions]) * [1, 1, 0.2, 1])))
			if not USE_ANALOG:
				for k in range(ACTIONS):
					actions_map[k, i, int(action_batch[i][k])] = 1.0
			else:
				actions_map[:, i, 0] = action_batch[i]
		feed = {model_nn.Y: y_batch, model_nn.X: initial_sensor_batch}
		for i in range(ACTIONS):
			feed[model_nn.actions[i]] = actions_map[i, :, :]
		loss, _ = sess.run([model_nn.cost, model_nn.optimizer], feed_dict = feed)
		print('step: ', time_elapsed, ', loss: ', loss)
	time_elapsed += 1
	display.add(time_elapsed, loss)
	plt.pause(0.001)
	model_nn.initial_sensor = stack_sensors
	if (time_elapsed + 1) % 1000 == 0:
		print('step: ', time_elapsed)
	if time_elapsed % 10000 == 0:
		print('checkpoint saved')
		saver.save(sess, os.getcwd() + "/model.ckpt")
		
# you must replace your own socket-io server
socketIO = SocketIO('https://huseinzol05.dynamic-dns.net', 9001)
car_namespace = socketIO.define(BaseNamespace, '/carsystem')
car_namespace.on('carsensor', on_car_response)
socketIO.wait()