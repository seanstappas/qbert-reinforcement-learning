#!/usr/bin/env python
# python_example.py
# Author: Ben Goodrich
#
# This is a direct port to python of the shared library example from
# ALE provided in doc/examples/sharedLibraryInterfaceExample.cpp
import sys
from random import randrange
from ale_python_interface import ALEInterface

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


def play_random_agent():
    ale = ALEInterface()

    # Get & Set the desired settings
    ale.setInt(b'random_seed', 123)

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
    print('Legal actions: {}'.format(legal_actions))

    # Play 10 episodes
    for episode in range(10):
        total_reward = 0
        while not ale.game_over():
            a = legal_actions[randrange(len(legal_actions))]
            # Apply an action and get the resulting reward
            reward = ale.act(a)
            if reward > 0:
                print('Chosen action: {}, reward: {}'.format(ACTIONS[a], reward))
            total_reward += reward
        print('Episode %d ended with score: %d' % (episode, total_reward))
        ale.reset_game()


if __name__ == '__main__':
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
