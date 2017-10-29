#!/usr/bin/env python
# python_example.py
# Author: Ben Goodrich
#
# This is a direct port to python of the shared library example from
# ALE provided in doc/examples/sharedLibraryInterfaceExample.cpp
import logging
import sys
import numpy as np
from random import randrange
from ale_python_interface import ALEInterface
import matplotlib.pyplot as plt

from main import setup_logging
from map import World

BLOCK_POSITIONS = [
    (38, 77),
    (66, 65), (66, 93),
    (95, 53), (95, 77), (95, 105),
    (124, 42), (124, 65), (124, 93), (124, 118),
    (153, 30), (153, 53), (153, 77), (153, 105), (153, 130),
    (182, 18), (182, 42), (182, 65), (182, 93), (182, 118), (182, 142),
]

SCORE_Y, SCORE_X = (10, 70)
INITIAL_COLOR = 210, 210, 64  # Yellow

NUM_EPISODES = 10
USE_SDL = True
ACTIONS = [
    'noop',
    'fire',
    'up',
    'right',
    'left',
    'down',
    'up-right',
    'up-left',
    'down-right',
    'down-left',
    'up-fire',
    'right-fire',
    'left-fire',
    'down-fire',
    'up-right-fire',
    'up-left-fire',
    'down-right-fire',
    'down-left-fire'
]


# Minimal actions: ['noop', 'fire', 'up', 'right', 'left', 'down']

def play_random_agent():
    ale = ALEInterface()

    # Get & Set the desired settings
    ale.setInt('random_seed', 123)
    ale.setInt('frame_skip', 60)
    ale.setFloat('repeat_action_probability', 0)

    # Set USE_SDL to true to display the screen. ALE must be compilied
    # with SDL enabled for this to work. On OSX, pygame init is used to
    # proxy-call SDL_main.
    if USE_SDL:
        if sys.platform == 'darwin':
            import pygame

            pygame.init()
            ale.setBool('sound', True)  # Sound doesn't work on OSX
        elif sys.platform.startswith('linux'):
            ale.setBool('sound', True)
        ale.setBool('display_screen', True)

    # Load the ROM file
    ale.loadROM('qbert.bin')

    # Get the list of legal actions
    legal_actions = ale.getLegalActionSet()
    minimal_actions = ale.getMinimalActionSet()
    logging.debug('Legal actions: {}'.format([ACTIONS[i] for i in legal_actions]))
    logging.debug('Minimal actions: {}'.format([ACTIONS[i] for i in minimal_actions]))
    # np.set_printoptions(threshold='nan')

    # Play 10 episodes
    width, height = ale.getScreenDims()
    rgb_screen = np.empty([height, width, 3], dtype=np.uint8)
    world = World(rgb_screen, ale)
    for episode in range(NUM_EPISODES):
        total_reward = 0
        while not ale.game_over():
            legal_actions = world.valid_action_numbers()
            a = legal_actions[randrange(len(legal_actions))]
            reward = ale.act(a)
            logging.debug('Frame number {}'.format(ale.getFrameNumber()))
            logging.debug('Chosen action: {}, reward: {}'.format(ACTIONS[a], reward))
            ale.getScreenRGB(rgb_screen)
            world.update(a)
            logging.debug('Desired color: {}'.format(world.desired_color))
            logging.debug('Current row: {}'.format(world.current_row))
            logging.debug('Current col: {}'.format(world.current_col))
            logging.debug('Desired colors: {}'.format(world.desired_colors))
            total_reward += reward

            plt.imshow(rgb_screen)
            plt.show()
        print('Episode %d ended with score: %d' % (episode, total_reward))
        ale.reset_game()


if __name__ == '__main__':
    setup_logging('debug')
    play_random_agent()

    # TODO: see and select actions on every kth frame: recommended every 4th frame

    # TODO: OR act at 12 steps/second (frame skip=5 within the stellarc configuration file.)

    # TODO: construct feature set (Basic or RAM best for Qbert, as shown in Bellemare et al.)
    # Aaron et al.: Tile coding is the most practical feature extraction technique. We also experimented
    # with convolutional features, where a set of predefined filters were run over the image each
    # step. The large number of convolutions required was too slow, at least using OpenCV2 or
    # Theano3 convolutional codes.
    # We performed our experiments using a variant of the BASIC representation, limited
    # to the SECAM color set. This representation is simply an encoding of the screen with a
    # courser grid, with a resolution of 14x16. Colors that occur in each 15x10 block are encoded
    # using indicator features, 1 for each of the 8 SECAM colors. Background subtraction is used
    # before encoding, as detailed in Bellemare et al. (2013).

    # TODO: Choose algorithm (SARSA, Q-learning) (Q-l. better if death can occur by exploration with e-greedy: Qbert!)
    # ETTR method performed best for Qbert (minimize expected time to next positive reward)
    # ^ Aaron et al.

    # Guo: no actions can change state of game while falling from cubes (can ignore these states if possible)

    # Probable best choice: Basic + Q-learning

    # TODO: Use pickle to save parameter weights

    # TODO: Only consider left, right, up, down actions

    # Human high scores: 15825, 27000
