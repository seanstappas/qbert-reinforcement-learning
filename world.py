import logging
from abc import ABCMeta, abstractmethod

import numpy as np
import sys
from ale_python_interface import ALEInterface

from actions import get_valid_actions, action_name_to_number, get_action_diffs, action_number_to_name, \
    get_action_number_diffs, get_valid_action_numbers

NUM_ROWS = 6
NUM_COLS = 6

QBERT_Y, QBERT_X = 28, 77
SCORE_Y, SCORE_X = 10, 70
BLOCK_COORDINATES = [
    [(38, 77)],
    [(66, 65), (66, 93)],
    [(95, 53), (95, 77), (95, 105)],
    [(124, 42), (124, 65), (124, 93), (124, 118)],
    [(153, 30), (153, 53), (153, 77), (153, 105), (153, 130)],
    [(182, 18), (182, 42), (182, 65), (182, 93), (182, 118), (182, 142)],
]  # (y, x) coordinates of blocks in RGB numpy array

INITIAL_DESIRED_COLORS = [
    [0],
    [0, 0],
    [0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
]  # Indicates if the desired colors are obtained at a block position

AGENT_POSITIONS = [
    [0],
    [0, 0],
    [0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
]  # Indicates if another agent is present at a block position (-1: purple, 0: none, +1: green)

LEFT_EDGE_BLOCKS = [(1, 0), (2, 0), (3, 0), (4, 0)]
RIGHT_EDGE_BLOCKS = [(1, 1), (2, 2), (3, 3), (4, 4)]
BOTTOM_BLOCKS = [(5, 1), (5, 2), (5, 3), (5, 4)]

COLOR_YELLOW = 210, 210, 64
COLOR_BLACK = 0, 0, 0
COLOR_QBERT = 181, 83, 40
COLOR_GREEN = 50, 132, 50
COLOR_PURPLE = 146, 70, 192

AGENT_BLOCK_OFFSET = -10


class World:
    __metaclass__ = ABCMeta

    @abstractmethod
    def to_state(self):
        raise NotImplementedError

    @abstractmethod
    def perform_action(self, a):
        raise NotImplementedError

    @abstractmethod
    def valid_actions(self):
        raise NotImplementedError


class QbertWorld(World):
    def __init__(self, rgb_screen, ale):
        self.ale = ale
        self.lives = ale.lives()
        self.rgb_screen = rgb_screen
        self.desired_color = COLOR_YELLOW
        self.desired_colors = INITIAL_DESIRED_COLORS
        self.agents = AGENT_POSITIONS
        self.current_row, self.current_col = 0, 0
        self.level = 1

    def to_state(self):
        current_position = self.current_row, self.current_col
        colors = list_to_tuple(self.desired_colors)
        return current_position, colors

    def perform_action(self, a):
        action = action_number_to_name(a)
        new_row, new_col = self.result_position(action)
        logging.debug('Waiting for Qbert to actually move to ({},{})'.format(new_row, new_col))
        rgb_y, rgb_x = BLOCK_COORDINATES[new_row][new_col]
        reward_sum = 0
        while not np.array_equal(self.rgb_screen[rgb_y + AGENT_BLOCK_OFFSET][rgb_x], COLOR_QBERT):
            if self.ale.lives() == 0:
                self.reset_position()
                return reward_sum
            reward_sum += self.ale.act(a)
            self.ale.getScreenRGB(self.rgb_screen)

        self.ale.getScreenRGB(self.rgb_screen)
        self.update_position(action)
        self.update_colors()
        return reward_sum

    def valid_actions(self):
        return get_valid_actions(self.current_row, self.current_col)

    def valid_action_numbers(self):
        valid_actions = self.valid_actions()
        return [action_name_to_number(a) for a in valid_actions]

    def result_position(self, action):
        diff_row, diff_col = get_action_diffs(action)
        return self.current_row + diff_row, self.current_col + diff_col

    def perform_action_name(self, action):
        a = action_name_to_number(action)
        new_row, new_col = self.result_position(action)
        logging.debug('Waiting for Qbert to actually move to ({},{})'.format(new_row, new_col))
        rgb_y, rgb_x = BLOCK_COORDINATES[new_row][new_col]
        reward_sum = 0
        while not np.array_equal(self.rgb_screen[rgb_y + AGENT_BLOCK_OFFSET][rgb_x], COLOR_QBERT):
            if self.ale.lives() == 0:
                self.reset_position()
                return reward_sum
            reward_sum += self.ale.act(a)
            self.ale.getScreenRGB(self.rgb_screen)

        self.ale.getScreenRGB(self.rgb_screen)
        self.update_position(action)
        self.update_colors()  # TODO: properly identify next level
        return reward_sum

    def update_colors(self):
        score_color = self.rgb_screen[SCORE_Y][SCORE_X]
        if not np.array_equal(score_color, COLOR_BLACK):
            self.desired_color = score_color
        level_won = True
        for row in range(NUM_ROWS):
            for col in range(row + 1):
                rgb_y, rgb_x = BLOCK_COORDINATES[row][col]
                # Evaluate color of block
                if np.array_equal(self.rgb_screen[rgb_y][rgb_x], self.desired_color):
                    self.desired_colors[row][col] = 1
                else:
                    self.desired_colors[row][col] = 0
                    level_won = False

                # Evaluate color of possible agents on blocks
                if np.array_equal(self.rgb_screen[rgb_y + AGENT_BLOCK_OFFSET][rgb_x], COLOR_PURPLE):
                    # Enemy (purple)
                    self.agents[row][col] = -1
                elif np.array_equal(self.rgb_screen[rgb_y + AGENT_BLOCK_OFFSET][rgb_x], COLOR_GREEN):
                    # Friendly (green)
                    self.agents[row][col] = 1
                else:
                    # No agent detected
                    self.agents[row][col] = 0
                    level_won = False
        if level_won:
            self.level += 1
            self.reset_position()

    def update_position(self, a):
        self.current_row, self.current_col = self.result_position(a)

    def reset_position(self):
        self.current_col, self.current_row = 0, 0

    def get_next_state(self, a):
        diff_row, diff_col = get_action_number_diffs(a)
        new_position = self.current_row + diff_row, self.current_col + diff_col
        new_colors = list_to_tuple_with_value(self.desired_colors, new_position[0], new_position[1], 1)
        return new_position, new_colors

    def get_close_states(self):
        states = []
        for a in get_valid_action_numbers(self.current_row, self.current_col):
            states.append(self.get_next_state(a))
        return states

        # TODO: Keep track of discs


def list_to_tuple(lst):
    return tuple(tuple(x for x in row) for row in lst)


def list_to_tuple_with_value(lst, row_num, col_num, val):
    return tuple(tuple(x if i != row_num or j != col_num else val for j, x in enumerate(row))
                 for i, row in enumerate(lst))


def setup_world(random_seed=123, frame_skip=1, repeat_action_probability=0, sound=True, display_screen=False):
    ale = ALEInterface()

    # Get & Set the desired settings
    ale.setInt('random_seed', random_seed)
    ale.setInt('frame_skip', frame_skip)
    ale.setFloat('repeat_action_probability', repeat_action_probability)

    if display_screen:
        if sys.platform == 'darwin':
            import pygame
            pygame.init()
        ale.setBool('sound', sound)
        
    ale.setBool('display_screen', display_screen)

    # Load the ROM file
    ale.loadROM('qbert.bin')

    # Get the list of legal actions
    legal_actions = ale.getLegalActionSet()
    minimal_actions = ale.getMinimalActionSet()
    logging.debug('Legal actions: {}'.format([action_number_to_name(a) for a in legal_actions]))
    logging.debug('Minimal actions: {}'.format([action_number_to_name(a) for a in minimal_actions]))

    width, height = ale.getScreenDims()
    rgb_screen = np.empty([height, width, 3], dtype=np.uint8)
    logging.debug('Waiting for Qbert to get into position...')
    while not np.array_equal(rgb_screen[QBERT_Y][QBERT_X], COLOR_QBERT):
        ale.act(0)
        ale.getScreenRGB(rgb_screen)
    logging.debug('Qbert in position!')
    world = QbertWorld(rgb_screen, ale)
    return world
