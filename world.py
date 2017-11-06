import logging
import sys
from abc import ABCMeta, abstractmethod

import numpy as np
from ale_python_interface import ALEInterface

from actions import get_action_diffs, action_number_to_name, \
    get_action_number_diffs, get_valid_action_numbers, get_inverse_action, ACTION_NUM_DIFFS, ACTION_NUM_DIFFS_WITH_NOOP

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

AGENT_BLOCK_OFFSET = -5
AGENT_BLOCK_OFFSET_RANGE = 30

DISC_OFFSET_Y = 14
DISC_OFFSET_X = 14

NO_OP = 0

SAM_SCORE = 300
GREEN_BALL_SCORE = 100
KILL_COILY_SCORE = 500
LOSE_LIFE_PENALTY = -100

LEVEL_BYTE = 99

FLASH_CHECK_Y, FLASH_CHECK_X = 40, 140


class World:
    __metaclass__ = ABCMeta

    @abstractmethod
    def perform_action(self, a):
        raise NotImplementedError


class QbertWorld(World):
    def __init__(self, random_seed, frame_skip, repeat_action_probability, sound, display_screen, state_repr):
        ale = ALEInterface()

        # Get & Set the desired settings
        if random_seed is not None:
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
        self.state_repr = state_repr

    def to_state_blocks(self):
        if self.state_repr is 'simple':
            return self.to_state_blocks_simple()
        elif self.state_repr is 'adjacent' or self.state_repr is 'adjacent_conservative':
            return self.to_state_blocks_adjacent()
        elif self.state_repr is 'verbose':
            return self.to_state_blocks_verbose()

    def to_state_enemies(self):
        if self.state_repr is 'simple':
            return self.to_state_enemies_simple()
        elif self.state_repr is 'adjacent':
            return self.to_state_enemies_adjacent()
        elif self.state_repr is 'adjacent_conservative':
            return self.to_state_enemies_adjacent_conservative()
        elif self.state_repr is 'verbose':
            return self.to_state_enemies_verbose()

    def to_state_friendlies(self):
        if self.state_repr is 'simple':
            return self.to_state_friendlies_simple()
        elif self.state_repr is 'adjacent' or self.state_repr is 'adjacent_conservative':
            return self.to_state_friendlies_simple()  # TODO: Make adjacent version of friendlies
        elif self.state_repr is 'verbose':
            return self.to_state_friendlies_verbose()

    def to_state_blocks_simple(self):
        """
        Simple state representation for blocks around Qbert.

        None: unattainable
        0: uncolored block
        1: colored block
        """
        row, col = self.current_row, self.current_col
        top_left = None
        top_right = None
        bot_left = None
        bot_right = None
        if col != 0:
            top_left = self.block_colors[row - 1][col - 1]
        if col != row:
            top_right = self.block_colors[row - 1][col]
        if row != NUM_ROWS - 1:
            bot_left = self.block_colors[row + 1][col]
            bot_right = self.block_colors[row + 1][col + 1]
        return top_left, top_right, bot_left, bot_right

    def to_state_blocks_adjacent(self):
        """
        Simple state representation for blocks around Qbert.

        None: unattainable
        x: number of adjacent uncolored blocks, including current block (0, 1, 2, 3 or 4)
        """
        row, col = self.current_row, self.current_col
        top_left = None
        top_right = None
        bot_left = None
        bot_right = None
        if col != 0:
            top_left = self.num_adjacent_uncolored_blocks(row - 1, col - 1)
        if col != row:
            top_right = self.num_adjacent_uncolored_blocks(row - 1, col)
        if row != NUM_ROWS - 1:
            bot_left = self.num_adjacent_uncolored_blocks(row + 1, col)
            bot_right = self.num_adjacent_uncolored_blocks(row + 1, col + 1)
        return top_left, top_right, bot_left, bot_right

    def num_adjacent_uncolored_blocks(self, row, col):
        num_adjacent = 0
        for diff_row, diff_col in ACTION_NUM_DIFFS_WITH_NOOP.values():
            r = row + diff_row
            c = col + diff_col
            if 0 <= r < NUM_ROWS and 0 <= c <= r and self.block_colors[r][c] == 0:
                num_adjacent += 1
        return num_adjacent

    def to_state_blocks_adjacent_old(self):
        """
        Simple state representation for blocks around Qbert.

        None: unattainable
        0: uncolored block
        1: colored block
        2: adjacent uncolored block
        """
        row, col = self.current_row, self.current_col
        top_left = None
        top_right = None
        bot_left = None
        bot_right = None
        if col != 0:
            top_left = self.adjacent_block_value(row - 1, col - 1)
        if col != row:
            top_right = self.adjacent_block_value(row - 1, col)
        if row != NUM_ROWS - 1:
            bot_left = self.adjacent_block_value(row + 1, col)
            bot_right = self.adjacent_block_value(row + 1, col + 1)
        return top_left, top_right, bot_left, bot_right

    def adjacent_block_value(self, row, col):
        if self.block_colors[row][col] == 0:
            return 0
        else:
            if self.is_adjacent_uncolored_block(row, col):
                return 2
            else:
                return 1

    def is_adjacent_uncolored_block(self, row, col):
        for diff_row, diff_col in ACTION_NUM_DIFFS.values():
            r = row + diff_row
            c = col + diff_col
            if 0 <= r < NUM_ROWS and 0 <= c <= r and self.block_colors[r][c] == 0:
                return True
        return False

    def to_state_blocks_verbose(self):
        if self.state_repr is 'simple':
            return self.to_state_blocks_simple()
        current_position = self.current_row, self.current_col
        logging.debug('Current position: {}'.format(current_position))
        colors = list_to_tuple(self.block_colors)
        return current_position, colors

    def to_state_enemies_simple(self):
        """
        Simple state representation for enemies around Qbert.

        None: unattainable
        0: block/disc
        1: enemy
        """
        row, col = self.current_row, self.current_col
        top_left = None
        top_right = None
        bot_left = None
        bot_right = None

        if col != 0 and self.enemies[row - 1][col - 1] == 0 or col == 0 and self.discs[row][0] == 1:
            top_left = 0
        elif col != 0 and self.enemies[row - 1][col - 1] == 1:
            top_left = 1

        if col != row and self.enemies[row - 1][col] == 0 or col == row and self.discs[row][1] == 1:
            top_right = 0
        elif col != row and self.enemies[row - 1][col] == 1:
            top_right = 1

        if row != NUM_ROWS - 1:
            if self.enemies[row + 1][col] == 0:
                bot_left = 0
            else:
                bot_left = 1
            if self.enemies[row + 1][col + 1] == 0:
                bot_right = 0
            else:
                bot_right = 1
        return top_left, top_right, bot_left, bot_right

    def to_state_enemies_adjacent(self):
        """
        Adjacent state representation for enemies around Qbert.

        None: unattainable/enemy
        0: block
        1: disc
        2: enemy adjacent
        """
        row, col = self.current_row, self.current_col
        top_left = None
        top_right = None
        bot_left = None
        bot_right = None

        if self.is_enemy_adjacent(row - 1, col - 1):
            top_left = 2
        elif col != 0 and self.enemies[row - 1][col - 1] == 0:
            top_left = 0
        elif col == 0 and self.discs[row][0] == 1:
            top_left = 1
            # TODO: Only go to discs when Coily is here, not purple ball.. (coily position in 0x27 and 0x45)

        if self.is_enemy_adjacent(row - 1, col):
            top_right = 2
        elif col != row and self.enemies[row - 1][col] == 0:
            top_right = 0
        elif col == row and self.discs[row][1] == 1:
            top_right = 1

        if row != NUM_ROWS - 1:
            if self.is_enemy_adjacent(row + 1, col):
                bot_left = 2
            elif self.enemies[row + 1][col] == 0:
                bot_left = 0

            if self.is_enemy_adjacent(row + 1, col + 1):
                bot_right = 2
            elif self.enemies[row + 1][col + 1] == 0:
                bot_right = 0
        return top_left, top_right, bot_left, bot_right

    def to_state_enemies_adjacent_conservative(self):
        """
        Adjacent state representation for enemies around Qbert.

        None: unattainable/enemy/enemy adjacent
        0: block
        1: disc
        """
        row, col = self.current_row, self.current_col
        top_left = None
        top_right = None
        bot_left = None
        bot_right = None

        if self.is_enemy_adjacent(row - 1, col - 1):
            top_left = None
        elif col != 0 and self.enemies[row - 1][col - 1] == 0:
            top_left = 0
        elif col == 0 and self.discs[row][0] == 1:
            top_left = 1
            # TODO: Only go to discs when Coily is here, not purple ball.. (coily position in 0x27 and 0x45)

        if self.is_enemy_adjacent(row - 1, col):
            top_right = None
        elif col != row and self.enemies[row - 1][col] == 0:
            top_right = 0
        elif col == row and self.discs[row][1] == 1:
            top_right = 1

        if row != NUM_ROWS - 1:
            if self.is_enemy_adjacent(row + 1, col):
                bot_left = None
            elif self.enemies[row + 1][col] == 0:
                bot_left = 0

            if self.is_enemy_adjacent(row + 1, col + 1):
                bot_right = None
            elif self.enemies[row + 1][col + 1] == 0:
                bot_right = 0
        return top_left, top_right, bot_left, bot_right

    def is_enemy_adjacent(self, row, col):
        if 0 <= row < NUM_ROWS and 0 <= col <= row and self.enemies[row][col] != 1:
            for diff_row, diff_col in ACTION_NUM_DIFFS.values():
                r = row + diff_row
                c = col + diff_col
                if 0 <= r < NUM_ROWS and 0 <= c <= r and self.enemies[r][c] == 1:
                    return True
        return False

    def is_friendly_adjacent(self, row, col):
        if 0 <= row < NUM_ROWS and 0 <= col <= row and self.friendlies[row][col] != 1:
            for diff_row, diff_col in ACTION_NUM_DIFFS.values():
                r = row + diff_row
                c = col + diff_col
                if 0 <= r < NUM_ROWS and 0 <= c <= r and self.friendlies[r][c] == 1:
                    return True
        return False

    def is_enemy_nearby(self):
        r, c = self.current_row, self.current_col
        return self.is_enemy_adjacent(r, c) or self.is_enemy_adjacent(r - 1, c - 1) or self.is_enemy_adjacent(r-1, c) \
               or self.is_enemy_adjacent(r + 1, c) or self.is_enemy_adjacent(r + 1, c + 1)

    def is_friendly_nearby(self):
        r, c = self.current_row, self.current_col
        return self.is_friendly_adjacent(r, c) or self.is_friendly_adjacent(r - 1, c - 1) \
               or self.is_friendly_adjacent(r-1, c) or self.is_friendly_adjacent(r + 1, c) \
               or self.is_friendly_adjacent(r + 1, c + 1)

    def to_state_enemies_verbose(self):
        current_position = self.current_row, self.current_col
        enemies = list_to_tuple(self.enemies)
        return current_position, enemies

    def to_state_friendlies_simple(self):
        """
        Simple state representation for green agents around Qbert.

        None: unattainable
        0: no green
        1: green
        """
        row, col = self.current_row, self.current_col
        top_left = None
        top_right = None
        bot_left = None
        bot_right = None
        if col != 0:
            top_left = self.friendlies[row - 1][col - 1]
        if col != row:
            top_right = self.friendlies[row - 1][col]
        if row != NUM_ROWS - 1:
            bot_left = self.friendlies[row + 1][col]
            bot_right = self.friendlies[row + 1][col + 1]
        return top_left, top_right, bot_left, bot_right

    def to_state_friendlies_verbose(self):
        current_position = self.current_row, self.current_col
        friendlies = list_to_tuple(self.friendlies)
        return current_position, friendlies

    def perform_action(self, a):
        score = 0
        friendly_score = 0
        enemy_score = 0
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
            if score_diff == SAM_SCORE:
                friendly_score = score_diff
            elif score_diff == KILL_COILY_SCORE:
                logging.info('Killed Coily!')
                enemy_score = score_diff
            else:
                score += score_diff
            self.ale.getRAM(self.ram)

        if self.ram[LEVEL_BYTE] + 1 != self.level:
            logging.info('Current level: {}'.format(self.level))
            self.level = self.ram[LEVEL_BYTE] + 1
            logging.info('Level won! Progressing to level {}'.format(self.level))
            score += self.reset_position()
        self.update_rgb()
        return score, friendly_score, enemy_score, enemy_penalty

    def result_position(self, action):
        diff_row, diff_col = get_action_diffs(action)
        return self.current_row + diff_row, self.current_col + diff_col

    def update_rgb(self):
        self.ale.getScreenRGB(self.rgb_screen)

        # Score
        score_color = self.rgb_screen[SCORE_Y][SCORE_X]
        if self.screen_not_flashing() \
                and not np.array_equal(score_color, COLOR_BLACK) \
                and not np.array_equal(score_color, self.desired_color):
            logging.debug('Identified {} as new desired color'.format(score_color))
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
                if self.screen_not_flashing():
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

    def screen_not_flashing(self):
        """
        Indicates the screen is flashing after a powerup.
        """
        return np.array_equal(self.rgb_screen[FLASH_CHECK_Y][FLASH_CHECK_X], COLOR_BLACK)

    def reset_position(self):
        reward = 0
        while not (self.ram[0] == 0 and self.ram[self.ram_size - 1] & 1):  # last bit = 1 and first byte = 0
            reward += self.ale.act(NO_OP)
            self.ale.getRAM(self.ram)
        self.update_rgb()
        return reward

    def reset(self):
        self.ale.getRAM(self.ram)
        self.level = self.ram[LEVEL_BYTE] + 1
        return self.reset_position()

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
