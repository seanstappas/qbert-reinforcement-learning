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
