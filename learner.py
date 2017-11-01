import logging
import random
from abc import ABCMeta

from actions import Actions

NUM_ROWS = 6
NUM_COLS = 6

INITIAL_PARAMETERS = [
    [0],
    [0, 0],
    [0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
]

ACTIONS_TO_NUMBERS = {
    'noop': 0,
    'up': 2,
    'right': 3,
    'left': 4,
    'down': 5
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

EPSILON = 0.1
N_e = 1
UNEXPLORED_REWARD = 5


class Learner:
    __metaclass__ = ABCMeta


class QLearner(Learner):
    def __init__(self, alpha, gamma):
        self.desired_color_parameters = INITIAL_PARAMETERS
        self.agent_parameters = INITIAL_PARAMETERS
        self.alpha = alpha
        self.gamma = gamma
        self.Q = {}
        self.N = {}

    def get_q(self, s, a):
        return self.Q.get((s, a), 0)

    def get_best_action_random(self, s):
        row, col = s[0]
        actions = Actions.get_valid_action_numbers(row, col)
        logging.debug('Valid actions: {}'.format([ACTIONS[a] for a in actions]))
        if random.random() < EPSILON:
            action = random.choice(actions)
            logging.debug('Randomly chose {}'.format(ACTIONS[action]))
            return action
        random.shuffle(actions)
        max_q = float('-inf')
        max_action = None
        for a in actions:
            if self.get_q(s, a) > max_q:
                max_q = self.get_q(s, a)
                max_action = a
        return max_action

    def get_best_action_optimistic(self, s):
        row, col = s[0]
        actions = Actions.get_valid_action_numbers(row, col)
        random.shuffle(actions)
        logging.debug('Valid actions: {}'.format([ACTIONS[a] for a in actions]))
        max_q = float('-inf')
        max_action = None
        for a in actions:
            q = UNEXPLORED_REWARD if self.N.get((s, a), 0) < N_e else self.get_q(s, a)
            if q > max_q:
                max_q = q
                max_action = a
        return max_action

    def get_max_q(self, s):
        max_q = float('-inf')
        row, col = s[0]
        for a in Actions.get_valid_action_numbers(row, col):
            max_q = max(max_q, self.Q.get((s, a), 0))
        return max_q

    def q_update_optimistic(self, s, a, s_next, reward, close_states):
        old_q = self.get_q(s, a)
        new_q = old_q + self.alpha * self.N.get((s, a), 0) * (reward + self.gamma * self.get_max_q(s_next) - old_q)
        self.Q[s, a] = new_q
        self.N[s, a] = self.N.get((s, a), 0) + 1
        for s_close in close_states:
            self.Q[s_close, a] = new_q

    def q_update(self, s, a, s_next, reward, close_states):
        old_q = self.get_q(s, a)
        new_q = old_q + self.alpha * (reward + self.gamma * self.get_max_q(s_next) - old_q)
        self.Q[s, a] = new_q
        for s_close in close_states:
            self.Q[s_close, a] = new_q

    def utility(self, world):
        utility = 0
        # Desired color utility
        for row in range(NUM_ROWS):
            for col in range(row + 1):
                utility += self.desired_color_parameters[row][col] * world.desired_colors[row][col]

        # Agent utility
        for row in range(NUM_ROWS):
            for col in range(row + 1):
                utility += self.agent_parameters[row][col] * world.agents[row][col]

        return utility

        # TODO: Implement subsumption: only learn from enemies when they are present on the board, and enemies take priority
