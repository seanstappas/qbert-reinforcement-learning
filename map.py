import numpy as np

NUM_ROWS = 6
NUM_COLS = 6

SCORE_Y, SCORE_X = (10, 70)
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

LEFT_EDGE_BLOCKS = [(1, 0), (2, 0), (3, 0), (4, 0)]
RIGHT_EDGE_BLOCKS = [(1, 1), (2, 2), (3, 3), (4, 4)]
BOTTOM_BLOCKS = [(5, 1), (5, 2), (5, 3), (5, 4)]

ACTION_DIFFS = {
    'noop': (0, 0),
    'up': (-1, 0),
    'right': (1, 1),
    'left': (-1, -1),
    'down': (1, 0)
}

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

COLOR_YELLOW = 210, 210, 64  # Yellow
COLOR_BLACK = 0, 0, 0


class World:
    def __init__(self, rgb_screen, ale):
        self.ale = ale
        self.lives = ale.lives()
        self.rgb_screen = rgb_screen
        self.desired_color = COLOR_YELLOW
        self.desired_colors = INITIAL_DESIRED_COLORS
        self.current_row, self.current_col = 0, 0

    def valid_actions(self):
        if (self.current_row, self.current_col) == (0, 0):
            return ['noop', 'right', 'down']
        elif (self.current_row, self.current_col) == (5, 0):
            return ['noop', 'right', 'down']
        elif (self.current_row, self.current_col) == (5, 5):
            return ['noop', 'right', 'down']
        elif (self.current_row, self.current_col) in BOTTOM_BLOCKS:
            return ['noop', 'left', 'up']
        elif (self.current_row, self.current_col) in LEFT_EDGE_BLOCKS:
            return ['noop', 'up', 'right', 'down']
        elif (self.current_row, self.current_col) in RIGHT_EDGE_BLOCKS:
            return ['noop', 'right', 'left', 'down']
        else:
            return ['noop', 'up', 'right', 'left', 'down']

    def valid_action_numbers(self):
        valid_actions = self.valid_actions()
        return [ACTIONS.index(a) for a in valid_actions]

    def result_position(self, action):
        diff_row, diff_col = ACTION_DIFFS[action]
        return self.current_row + diff_row, self.current_col + diff_col

    def update(self, action):
        if self.ale.lives() != self.lives:
            self.lives = self.ale.lives()
            self.current_row, self.current_col = 0, 0
        else:
            a = ACTIONS[action]
            self.update_position(a)
        self.update_colors()

    def update_colors(self):
        score_color = self.rgb_screen[SCORE_Y][SCORE_X]
        if not np.array_equal(score_color, COLOR_BLACK):
            self.desired_color = score_color
        for row in range(NUM_ROWS):
            for col in range(row + 1):
                rgb_y, rgb_x = BLOCK_COORDINATES[row][col]
                if np.array_equal(self.rgb_screen[rgb_y][rgb_x], self.desired_color):
                    self.desired_colors[row][col] = 1
                else:
                    self.desired_colors[row][col] = 0

    def update_position(self, action):
        self.current_row, self.current_col = self.result_position(action)
