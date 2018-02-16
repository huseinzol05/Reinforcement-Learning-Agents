__author__ = 'justinarmstrong'

import os
import pygame as pg
from . import model
from . import realtime
import tensorflow as tf
import numpy as np
import random
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
sns.set()

time_elapsed, loss = 0, 0
sess = tf.InteractiveSession()
deepq = model.Model()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(tf.global_variables())
fig, axes = plt.subplots(figsize = (3, 3))
display = realtime.RealtimePlot(axes)

try:
    saver.restore(sess, os.getcwd() + "/model.ckpt")
    print("Done load checkpoint")
except:
    print ("start from fresh variables")

keybinding = {
    'action':pg.K_s,
    'jump':pg.K_a,
    'left':pg.K_LEFT,
    'right':pg.K_RIGHT,
    'down':pg.K_DOWN
}

keypress = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

def edit_keypress(argmax):
    string = ''
    # left
    if argmax == 0:
        string = 'go to left'
        keypress[keybinding['jump']] = 0
        keypress[keybinding['left']] = 1
        keypress[keybinding['right']] = 0
    # right
    if argmax == 1:
        string = 'go to right'
        keypress[keybinding['jump']] = 0
        keypress[keybinding['left']] = 0
        keypress[keybinding['right']] = 1
    # right + jump
    if argmax == 2:
        string = 'right and jump!'
        keypress[keybinding['jump']] = 1
        keypress[keybinding['left']] = 0
        keypress[keybinding['right']] = 1
    # left + jump
    if argmax == 3:
        string = 'left and jump!'
        keypress[keybinding['jump']] = 1
        keypress[keybinding['left']] = 1
        keypress[keybinding['right']] = 0
    # jump
    if argmax == 4:
        string = 'jump!'
        keypress[keybinding['jump']] = 1
        keypress[keybinding['left']] = 0
        keypress[keybinding['right']] = 0
    # do nothing
    if argmax == 5:
        string = 'do nothing..'
        keypress[keybinding['jump']] = 0
        keypress[keybinding['left']] = 0
        keypress[keybinding['right']] = 0
    return string


class Control(object):
    """Control class for entire project. Contains the game loop, and contains
    the event_loop which passes events to States as needed. Logic for flipping
    states is also found here."""
    def __init__(self, caption):
        self.screen = pg.display.get_surface()
        self.done = False
        self.clock = pg.time.Clock()
        self.caption = caption
        self.fps = 60
        self.show_fps = False
        self.current_time = 0.0
        self.keys = keypress
        self.state_dict = {}
        self.state_name = None
        self.state = None

    def setup_states(self, state_dict, start_state):
        self.state_dict = state_dict
        self.state_name = start_state
        self.state = self.state_dict[self.state_name]

    def update(self):
        global time_elapsed
        global epsilon
        global loss
        self.current_time = pg.time.get_ticks()
        if self.state.quit:
            self.done = True
        elif self.state.done:
            self.flip_state()
        return_tuple = self.state.update(self.screen, self.keys, self.current_time)
        if return_tuple is not None:
            if deepq.INITIAL_GAME:
                for i in range(deepq.INITIAL_IMAGES.shape[2]):
                    deepq.INITIAL_IMAGES[:, :, i] = return_tuple[0]
                deepq.INITIAL_GAME = False
            action = deepq.select_action(deepq.INITIAL_IMAGES)
            str_action = edit_keypress(action)
            return_tuple = self.state.update(self.screen, self.keys, self.current_time)
            reward = return_tuple[1]
            new_state = np.append(return_tuple[0].reshape([80, 80, 1]), deepq.INITIAL_IMAGES[:, :, :3], axis = 2)
            dead = return_tuple[2]
            deepq.memorize(deepq.INITIAL_IMAGES, action, reward, new_state, dead)
            if dead:
                print 'oh no, mario died!'
                deepq.INITIAL_GAME = True
            batch_size = min(len(deepq.MEMORIES), deepq.BATCH_SIZE)
            replay = random.sample(deepq.MEMORIES, batch_size)
            X, Y = deepq.construct_memories(replay)
            cost, _ = sess.run([deepq.cost, deepq.optimizer], feed_dict={deepq.X: X, deepq.Y:Y})
            deepq.INITIAL_IMAGES = new_state
            time_elapsed += 1
            print 'epoch %d, loss %f, mario action %s, mario reward %d' % (time_elapsed, cost, str_action, reward)
            display.add(time_elapsed, cost)
            plt.pause(0.0001)
            if (time_elapsed + 1) % 100000 == 0:
                print('checkpoint saved')
                saver.save(sess, os.getcwd() + "/model.ckpt")

    def flip_state(self):
        previous, self.state_name = self.state_name, self.state.next
        persist = self.state.cleanup()
        self.state = self.state_dict[self.state_name]
        self.state.startup(self.current_time, persist)
        self.state.previous = previous


    def event_loop(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.done = True
            elif event.type == pg.KEYDOWN:
                self.keys = pg.key.get_pressed()
                self.toggle_show_fps(event.key)
            elif event.type == pg.KEYUP:
                self.keys = pg.key.get_pressed()
            self.state.get_event(event)


    def toggle_show_fps(self, key):
        if key == pg.K_F5:
            self.show_fps = not self.show_fps
            if not self.show_fps:
                pg.display.set_caption(self.caption)


    def main(self):
        """Main loop for entire program"""
        while not self.done:
            self.event_loop()
            self.update()
            pg.display.update()
            self.clock.tick(self.fps)
            if self.show_fps:
                fps = self.clock.get_fps()
                with_fps = "{} - {:.2f} FPS".format(self.caption, fps)
                pg.display.set_caption(with_fps)


class _State(object):
    def __init__(self):
        self.start_time = 0.0
        self.current_time = 0.0
        self.done = False
        self.quit = False
        self.next = None
        self.previous = None
        self.persist = {}

    def get_event(self, event):
        pass

    def startup(self, current_time, persistant):
        self.persist = persistant
        self.start_time = current_time

    def cleanup(self):
        self.done = False
        return self.persist

    def update(self, surface, keys, current_time):
        pass



def load_all_gfx(directory, colorkey=(255,0,255), accept=('.png', 'jpg', 'bmp')):
    graphics = {}
    for pic in os.listdir(directory):
        name, ext = os.path.splitext(pic)
        if ext.lower() in accept:
            img = pg.image.load(os.path.join(directory, pic))
            if img.get_alpha():
                img = img.convert_alpha()
            else:
                img = img.convert()
                img.set_colorkey(colorkey)
            graphics[name]=img
    return graphics


def load_all_music(directory, accept=('.wav', '.mp3', '.ogg', '.mdi')):
    songs = {}
    for song in os.listdir(directory):
        name,ext = os.path.splitext(song)
        if ext.lower() in accept:
            songs[name] = os.path.join(directory, song)
    return songs


def load_all_fonts(directory, accept=('.ttf')):
    return load_all_music(directory, accept)


def load_all_sfx(directory, accept=('.wav','.mpe','.ogg','.mdi')):
    effects = {}
    for fx in os.listdir(directory):
        name, ext = os.path.splitext(fx)
        if ext.lower() in accept:
            effects[name] = pg.mixer.Sound(os.path.join(directory, fx))
    return effects
