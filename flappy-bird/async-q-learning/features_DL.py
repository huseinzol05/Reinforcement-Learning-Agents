import threading
import time
import tensorflow as tf
import numpy as np
import random
from ple import PLE
from ple.games.flappybird import FlappyBird

TMAX = 5000000
T = 0
copy_network = 10000
update_network = 32
THREADS = 12
WISHED_SCORE = 10

INPUTS = 8
ACTIONS = 2
GAMMA = 0.99
OBSERVE = 5
LAYER_SIZE = 500
EXPLORE = 400000
FINAL_EPSILONS = [0.01, 0.01, 0.05]
INITIAL_EPSILONS = [0.4, 0.3, 0.3]
EPSILONS = 3

def createNetwork():
    X = tf.placeholder(tf.float32, (None, INPUTS))
    input_layer = tf.Variable(tf.random_normal([INPUTS, LAYER_SIZE]))
    output_layer = tf.Variable(tf.random_normal([LAYER_SIZE, ACTIONS]))
    feed_forward = tf.nn.relu(tf.matmul(self.X, input_layer))
    logits = tf.matmul(feed_forward, output_layer)
    return X, input_layer, output_layer, logits

def copyTargetNetwork(sess):
    sess.run(copy_network)
    
def actorBird(num, sess, lock):
    global TMAX, T
    lock.acquire()
    game = FlappyBird(pipe_gap=125)
    env = PLE(game, fps=30, display_screen=True, force_fps=True)
    env.init()
    env.getGameState = self.game.getGameState
    lock.release()
    state_batch = []
    action_batch = []
    y_batch = []
    time.sleep(3*num)
    copyTargetNetwork(sess)
    epsilon_index = random.randrange(EPSILONS)
    INITIAL_EPSILON = INITIAL_EPSILONS[epsilon_index]
    FINAL_EPSILON =  FINAL_EPSILONS[epsilon_index]
    epsilon = INITIAL_EPSILON
    print("THREAD %d, EXPLORATION POLICY => INITIAL_EPSILON: %f, FINAL_EPSILON: %f"%(num,INITIAL_EPSILON, FINAL_EPSILON))
    t, score = 0, 0
    while T < TMAX and score < WISHED_SCORE:
        lock.acquire()
        self.env.reset_game()
        state = self.get_state()
        lock.release()
        done = False
        while not done:
            logits = G_logits.eval(sess, feed_dict={X:[state]})
            actions = np.zeros((ACTIONS))
            action_index = 0
            
            if random.random() <= epsilon or t <= OBSERVE:
                action_index = random.randrange(ACTIONS)
                actions[action_index] = 1
            else:
                action_index = np.argmax(logits)
                actions[action_index] = 1
            
            if epsilon > FINAL_EPSILON and t > OBSERVE:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
                
            real_action = 119 if action_index == 1 else None
            reward = self.env.act(real_action)
            lock.acquire()
            new_state = self.get_state()
            done = self.env.game_over()
            lock.release()
            logits_target = G_logits_target.eval(sess, feed_dict={X_target:[new_state]})
            if done:
                y_batch.append(reward)
            else:
                y_batch.append(r_t + GAMMA * np.max(logits_target))
            action_batch.append(actions)
            state_batch.append(state)
            state = new_state
            T += 1
            t += 1
            score += reward
            
            if T % copy_network == 0:
                copyTargetNetwork(sess)
            
            if t % update_network or done:
                if state_batch:
                    optimizer.run(sess, feed_dict={Y:y_batch, ACTION:action_batch,X:state_batch})
                action_batch = []
                y_batch = []
                state_batch = []
                
            if t % 5000 == 0:
                saver.save(sess, 'save_networks_asyn/' + GAME + '-dqn', global_step = t)
                
            state = ""
            if t <= OBSERVE:
                state = "observe"
            elif t > OBSERVE and t <= OBSERVE + EXPLORE:
                state = "explore"
            else:
                state = "train"
        
X, input_layer, output_layer, logits = createNetwork()
ACTION = tf.placeholder("float", [None, ACTIONS])
Y = tf.placeholder("float", [None])
reduce_logits_action = tf.reduce_sum(tf.mul(logits, ACTION), reduction_indices=1)
cost = tf.reduce_mean(tf.square(Y - reduce_logits_action))
optimizer = tf.train.RMSPropOptimizer(0.00025, 0.95, 0.95, 0.01).minimize(cost)
X_target, input_layer_target, output_layer_target, logits_target = createNetwork()

sess = tf.InteractiveSession()
saver = tf.train.Saver()
sess.run(tf.initialize_all_variables())
checkpoint = tf.train.get_checkpoint_state("save_networks_asyn")
if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(sess, checkpoint.model_checkpoint_path)
    print "Successfully loaded:", checkpoint.model_checkpoint_path
    
lock = threading.Lock()
threads = []
for i in range(THREADS):
    t = threading.Thread(target=actorBird, args=(i,sess, lock))
    threads.append(t)

for x in threads:
    x.start()

for x in threads:
    x.join()