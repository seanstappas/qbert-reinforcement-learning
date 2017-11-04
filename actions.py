LEFT_EDGE_BLOCKS = [(1, 0), (2, 0), (3, 0), (4, 0)]
RIGHT_EDGE_BLOCKS = [(1, 1), (2, 2), (3, 3), (4, 4)]
BOTTOM_BLOCKS = [(5, 1), (5, 2), (5, 3), (5, 4)]

ACTIONS_TO_NUMBERS = {
    'up': 2,
    'right': 3,
    'left': 4,
    'down': 5
}

ACTION_DIFFS = {
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

ACTION_NUM_DIFFS = {
    2: (-1, 0),
    3: (1, 1),
    4: (-1, -1),
    5: (1, 0)
}

INVERSE_ACTIONS = {
    2: 5,
    3: 4,
    4: 3,
    5: 2
}


def get_valid_action_numbers_from_state(s, state_repr='simple'):
    if state_repr is 'simple' or state_repr is 'adjacent':
        actions = []
        top_left, top_right, bot_left, bot_right = s
        if top_left is not None:
            actions.append(ACTIONS_TO_NUMBERS['left'])
        if top_right is not None:
            actions.append(ACTIONS_TO_NUMBERS['up'])
        if bot_left is not None:
            actions.append(ACTIONS_TO_NUMBERS['down'])
        if bot_right is not None:
            actions.append(ACTIONS_TO_NUMBERS['right'])
        return actions
    elif state_repr is 'verbose':
        row, col = s[0]
        return get_valid_action_numbers(row, col)
    else:
        return []


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


def action_number_to_name(a):
    return ACTIONS[a]


def action_name_to_number(action):
    return ACTIONS_TO_NUMBERS[action]


def get_action_diffs(action):
    return ACTION_DIFFS[action]


def get_action_number_diffs(a):
    return ACTION_NUM_DIFFS[a]


def get_inverse_action(a):
    return INVERSE_ACTIONS[a]
