

NUM_ROWS = 6
NUM_COLS = 6

LEFT_EDGE_BLOCKS = [(1, 0), (2, 0), (3, 0), (4, 0)]
RIGHT_EDGE_BLOCKS = [(1, 1), (2, 2), (3, 3), (4, 4)]
BOTTOM_BLOCKS = [(5, 1), (5, 2), (5, 3), (5, 4)]

ACTIONS_TO_NUMBERS = {
    'noop': 0,
    'up': 2,
    'right': 3,
    'left': 4,
    'down': 5
}

ACTION_DIFFS = {
    'noop': (0, 0),
    'up': (-1, 0),
    'right': (1, 1),
    'left': (-1, -1),
    'down': (1, 0)
}


class Actions:
    @staticmethod
    def get_valid_action_numbers(row, col):
        if (row, col) == (0, 0):
            return [3, 5]
        elif (row, col) == (5, 0):
            return [2]
        elif (row, col) == (5, 5):
            return [4]
        elif (row, col) in BOTTOM_BLOCKS:
            return [2, 4]
        elif (row, col) in LEFT_EDGE_BLOCKS:
            return [2, 3, 5]
        elif (row, col) in RIGHT_EDGE_BLOCKS:
            return [3, 4, 5]
        else:
            return [2, 3, 4, 5]

    @staticmethod
    def get_valid_actions(row, col):
        if (row, col) == (0, 0):
            return ['right', 'down']
        elif (row, col) == (5, 0):
            return ['up']
        elif (row, col) == (5, 5):
            return ['left']
        elif (row, col) in BOTTOM_BLOCKS:
            return ['left', 'up']
        elif (row, col) in LEFT_EDGE_BLOCKS:
            return ['up', 'right', 'down']
        elif (row, col) in RIGHT_EDGE_BLOCKS:
            return ['right', 'left', 'down']
        else:
            return ['up', 'right', 'left', 'down']
