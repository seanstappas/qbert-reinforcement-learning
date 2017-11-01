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

from agent import QbertAgent
from learner import QLearner
from main import setup_logging
from world import QbertWorld

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

COLOR_QBERT = 181, 83, 40


# Minimal actions: ['noop', 'fire', 'up', 'right', 'left', 'down']

def setup_world():
    ale = ALEInterface()

    # Get & Set the desired settings
    ale.setInt('random_seed', 123)
    ale.setInt('frame_skip', 0)
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
    logging.debug('Waiting for Qbert to get into position...')
    while not np.array_equal(rgb_screen[28][77], COLOR_QBERT):
        ale.act(0)
        ale.getScreenRGB(rgb_screen)
    logging.debug('Qbert in position!')
    world = QbertWorld(rgb_screen, ale)
    return world


def play_random_agent(world):
    for episode in range(NUM_EPISODES):
        total_reward = 0
        world.reset_position()
        while not world.ale.game_over():
            legal_actions = world.valid_action_numbers()
            a = legal_actions[randrange(len(legal_actions))]
            reward = world.perform_action(a)

            logging.debug('Chosen action: {}, reward: {}'.format(ACTIONS[a], reward))
            logging.debug('Current row/col: ({}, {}): '.format(world.current_row, world.current_col))
            logging.debug('Desired color: {}'.format(world.desired_color))
            logging.debug('Current row: {}'.format(world.current_row))
            logging.debug('Current col: {}'.format(world.current_col))
            logging.debug('Desired colors: {}'.format(world.desired_colors))
            logging.debug('Agents: {}'.format(world.agents))
            logging.debug('Reward: {}'.format(reward))

            total_reward += reward

            # plt.imshow(rgb_screen)
            # plt.show()
            # plt.savefig('report/screenshots/screenshot_{}'.format(i))
        print('Episode %d ended with score: %d' % (episode, total_reward))
        world.ale.reset_game()


ALPHA = 0.1
GAMMA = 0.9


def play_learning_agent(world):
    learner = QLearner(ALPHA, GAMMA)
    agent = QbertAgent(world, learner)
    for episode in range(NUM_EPISODES):
        total_reward = 0
        world.reset_position()
        while not world.ale.game_over():
            total_reward += agent.action()
        print('Episode %d ended with score: %d' % (episode, total_reward))
        world.ale.reset_game()


def play():
    world = setup_world()
    # play_random_agent(world)
    play_learning_agent(world)


if __name__ == '__main__':
    setup_logging('debug')
    play()

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
