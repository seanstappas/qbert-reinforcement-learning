import logging
from abc import ABCMeta, abstractmethod

import numpy as np
import sys
import matplotlib.pyplot as plt
from ale_python_interface import ALEInterface

from actions import get_valid_actions, action_name_to_number, get_action_diffs, action_number_to_name, \
    get_action_number_diffs, get_valid_action_numbers, get_inverse_action

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

INITIAL_COLORS = [
    [0],
    [0, 0],
    [0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
]  # Indicates if the desired colors are obtained at a block position (0 for starting, 1 for destination color)

INITIAL_ENEMY_POSITIONS = [
    [0],
    [0, 0],
    [0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
]  # Indicates if an enemy (purple) is present at a block position

INITIAL_FRIENDLY_POSITIONS = [
    [0],
    [0, 0],
    [0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
]  # Indicates if a friendly agent (green) is present at a block position

INITIAL_DISCS = [
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0]
]  # Indicates if there is a disc at the left or right at each row

LEFT_EDGE_BLOCKS = [(1, 0), (2, 0), (3, 0), (4, 0)]
RIGHT_EDGE_BLOCKS = [(1, 1), (2, 2), (3, 3), (4, 4)]
BOTTOM_BLOCKS = [(5, 1), (5, 2), (5, 3), (5, 4)]

COLOR_YELLOW = 210, 210, 64
COLOR_BLACK = 0, 0, 0
COLOR_QBERT = 181, 83, 40
COLOR_GREEN = 50, 132, 50
COLOR_PURPLE = 146, 70, 192

AGENT_BLOCK_OFFSET = -10
AGENT_BLOCK_OFFSET_RANGE = 10

DISC_OFFSET_Y = 14
DISC_OFFSET_X = 14

NO_OP = 0

SAM_SCORE = 300
GREEN_BALL_SCORE = 100
LOSE_LIFE_PENALTY = -100

LEVEL_WON_THRESHOLD = 1000

LEVEL_BYTE = 99


class World:
    __metaclass__ = ABCMeta

    @abstractmethod
    def perform_action(self, a):
        raise NotImplementedError

    @abstractmethod
    def valid_actions(self):
        raise NotImplementedError


class QbertWorld(World):
    def __init__(self, ale, rgb_screen, ram):
        # ALE components
        self.ale = ale
        self.lives = ale.lives()
        self.rgb_screen = rgb_screen
        self.ram_size = ale.getRAMSize()
        self.ram = ram

        # Verbose state representation
        self.desired_color = COLOR_YELLOW
        self.block_colors = INITIAL_COLORS
        self.enemies = INITIAL_ENEMY_POSITIONS
        self.friendlies = INITIAL_FRIENDLY_POSITIONS
        self.discs = INITIAL_DISCS
        self.current_row, self.current_col = 0, 0
        self.level = 1
        self.enemy_present = False
        self.friendly_present = False

    def to_simple_state_blocks(self):
        row, col = self.current_row, self.current_col
        top_left = None
        if col != 0:
            pass
        top_right = None
        if col != row:
            pass
        bot_left = None
        bot_right = None
        if row == NUM_ROWS - 1:
            pass
        return top_left, top_right, bot_left, bot_right

    def to_state_blocks(self):
        current_position = self.current_row, self.current_col
        logging.debug('Current position: {}'.format(current_position))
        colors = list_to_tuple(self.block_colors)
        return current_position, colors

    def to_state_enemies(self):
        current_position = self.current_row, self.current_col
        enemies = list_to_tuple(self.enemies)
        return current_position, enemies

    def to_state_friendlies(self):
        current_position = self.current_row, self.current_col
        friendlies = list_to_tuple(self.friendlies)
        return current_position, friendlies

    def perform_action(self, a):
        score = 0
        friendly_score = 0
        enemy_penalty = 0
        score += self.ale.act(a)
        initial_num_lives = self.ale.lives()
        self.ale.getRAM(self.ram)
        while not (self.ram[0] == 0 and self.ram[self.ram_size - 1] & 1):  # last bit = 1 and first byte = 0
            if self.ale.lives() < initial_num_lives:
                enemy_penalty = LOSE_LIFE_PENALTY
            if self.ale.lives() == 0:
                break
            score_diff = self.ale.act(NO_OP)
            if score_diff == SAM_SCORE or score_diff == GREEN_BALL_SCORE:
                friendly_score = score_diff
            score += score_diff
            self.ale.getRAM(self.ram)

        self.update()

        if self.ram[LEVEL_BYTE] + 1 != self.level:
            logging.info('Current level: {}'.format(self.level))
            self.level = self.ram[LEVEL_BYTE] + 1
            logging.info('Level won! Progressing to level {}'.format(self.level))
            self.reset_position()
        return score, friendly_score, enemy_penalty

    def valid_actions(self):
        return get_valid_actions(self.current_row, self.current_col)

    def valid_action_numbers(self):
        valid_actions = self.valid_actions()
        return [action_name_to_number(a) for a in valid_actions]

    def result_position(self, action):
        diff_row, diff_col = get_action_diffs(action)
        return self.current_row + diff_row, self.current_col + diff_col

    def update(self):
        self.ale.getScreenRGB(self.rgb_screen)

        # Score
        score_color = self.rgb_screen[SCORE_Y][SCORE_X]
        if not np.array_equal(score_color, COLOR_BLACK):
            self.desired_color = score_color

        self.enemy_present = False
        self.friendly_present = False
        for row in range(NUM_ROWS):
            for col in range(row + 1):
                rgb_y, rgb_x = BLOCK_COORDINATES[row][col]

                # Color of block
                if np.array_equal(self.rgb_screen[rgb_y][rgb_x], self.desired_color):
                    self.block_colors[row][col] = 1
                else:
                    self.block_colors[row][col] = 0

                self.enemies[row][col] = 0
                self.friendlies[row][col] = 0
                # Agents
                for y_offset in range(AGENT_BLOCK_OFFSET_RANGE):
                    agent_offset = AGENT_BLOCK_OFFSET - y_offset
                    # Enemy (purple)
                    if np.array_equal(self.rgb_screen[rgb_y + agent_offset][rgb_x], COLOR_PURPLE):
                        self.enemies[row][col] = 1
                        self.enemy_present = True

                    # Friendly (green)
                    if np.array_equal(self.rgb_screen[rgb_y + agent_offset][rgb_x], COLOR_GREEN):
                        self.friendlies[row][col] = 1
                        self.friendly_present = True

                    # Qbert (orange)
                    if np.array_equal(self.rgb_screen[rgb_y + agent_offset][rgb_x], COLOR_QBERT):
                        self.current_row, self.current_col = row, col

                # Discs (relative to edge blocks)
                if col == 0:
                    if np.array_equal(self.rgb_screen[rgb_y - DISC_OFFSET_Y][rgb_x - DISC_OFFSET_X], COLOR_BLACK):
                        self.discs[row][0] = 0
                    else:
                        self.discs[row][0] = 1
                if col == row:
                    if np.array_equal(self.rgb_screen[rgb_y - DISC_OFFSET_Y][rgb_x + DISC_OFFSET_X], COLOR_BLACK):
                        self.discs[row][1] = 0
                    else:
                        self.discs[row][1] = 1
        logging.debug('Discs: {}'.format(self.discs))

    def update_position(self, a):
        self.current_row, self.current_col = self.result_position(a)

    def reset_position(self):
        while not (self.ram[0] == 0 and self.ram[self.ram_size - 1] & 1):  # last bit = 1 and first byte = 0
            self.ale.act(NO_OP)
            self.ale.getRAM(self.ram)
        self.current_col, self.current_row = 0, 0

    def reset(self):
        self.ale.getRAM(self.ram)
        self.level = self.ram[LEVEL_BYTE] + 1
        self.reset_position()

    def get_next_state(self, a):
        diff_row, diff_col = get_action_number_diffs(a)
        new_position = self.current_row + diff_row, self.current_col + diff_col
        new_colors = list_to_tuple_with_value(self.block_colors, new_position[0], new_position[1], 1)
        return new_position, new_colors

    def get_close_states_actions(self, initial_action, distance_metric='simple'):
        states = []
        actions = []
        if distance_metric is 'simple':
            for a in get_valid_action_numbers(self.current_row, self.current_col):
                states.append(self.get_next_state(a))
                actions.append(initial_action)
        elif distance_metric is 'adjacent':
            for a in get_valid_action_numbers(self.current_row, self.current_col):
                states.append(self.get_next_state(a))
                actions.append(get_inverse_action(a))
        return states, actions


def list_to_tuple(lst):
    return tuple(tuple(x for x in row) for row in lst)


def list_to_tuple_with_value(lst, row_num, col_num, val):
    return tuple(tuple(x if i != row_num or j != col_num else val for j, x in enumerate(row))
                 for i, row in enumerate(lst))


def setup_world(random_seed=123, frame_skip=4, repeat_action_probability=0, sound=True, display_screen=False):
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

    ram_size = ale.getRAMSize()
    ram = np.zeros(ram_size, dtype=np.uint8)
    world = QbertWorld(ale, rgb_screen, ram)
    return world
