import logging
import random
from abc import ABCMeta, abstractmethod

from actions import get_valid_action_numbers, action_number_to_name


class Learner:
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_best_action(self, s):
        raise NotImplementedError

    @abstractmethod
    def update(self, s, a, s_next, reward):
        raise NotImplementedError


class QLearner(Learner):
    def __init__(self, world, alpha=0.1, gamma=0.9, epsilon=0.1, unexplored_threshold=1, unexplored_reward=5,
                 exploration='random', generalization='simple_distance'):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.unexplored_threshold = unexplored_threshold
        self.unexplored_reward = unexplored_reward
        self.exploration = exploration
        self.generalization = generalization
        self.Q = {}
        self.N = {}
        self.world = world

    def get_best_action(self, s):
        if self.exploration is 'optimistic':
            return self.get_best_action_optimistic(s)
        elif self.exploration is 'random':
            return self.get_best_action_random(s)
        else:
            return None

    def update(self, s, a, s_next, reward):
        if self.exploration is 'optimistic':
            self.q_update_optimistic(s, a, s_next, reward)
        elif self.exploration is 'random':
            self.q_update_random(s, a, s_next, reward)
        logging.debug('Q matrix: {}'.format(self.Q.values()))
        logging.debug('N matrix: {}'.format(self.N.values()))

    def get_q(self, s, a):
        return self.Q.get((s, a), 0)

    def get_best_action_random(self, s):
        row, col = s[0]
        actions = get_valid_action_numbers(row, col)
        logging.debug('Valid actions: {}'.format([action_number_to_name(a) for a in actions]))
        if random.random() < self.epsilon:
            action = random.choice(actions)
            logging.debug('Randomly chose {}'.format(action_number_to_name(action)))
            return action
        random.shuffle(actions)  # If equal values, will choose random one
        max_q = float('-inf')
        max_action = None
        for a in actions:
            if self.get_q(s, a) > max_q:
                max_q = self.get_q(s, a)
                max_action = a
        return max_action

    def get_best_action_optimistic(self, s):
        row, col = s[0]
        actions = get_valid_action_numbers(row, col)
        logging.debug('Valid actions: {}'.format([action_number_to_name(a) for a in actions]))
        max_q = float('-inf')
        max_action = None
        random.shuffle(actions)
        for a in actions:
            q = self.unexplored_reward if self.N.get((s, a), 0) < self.unexplored_threshold else self.get_q(s, a)
            if q > max_q:
                max_q = q
                max_action = a
        return max_action

    def get_max_q(self, s):
        max_q = float('-inf')
        row, col = s[0]
        for a in get_valid_action_numbers(row, col):
            max_q = max(max_q, self.Q.get((s, a), 0))
        return max_q

    def q_update_optimistic(self, s, a, s_next, reward):
        old_q = self.get_q(s, a)
        new_q = old_q + self.alpha * self.N.get((s, a), 0) * (reward + self.gamma * self.get_max_q(s_next) - old_q)
        self.Q[s, a] = new_q
        self.N[s, a] = self.N.get((s, a), 0) + 1
        if self.generalization is 'simple_distance':
            for s_close in self.world.get_close_states():
                self.Q[s_close, a] = new_q

    def q_update_random(self, s, a, s_next, reward):
        old_q = self.get_q(s, a)
        new_q = old_q + self.alpha * (reward + self.gamma * self.get_max_q(s_next) - old_q)
        self.Q[s, a] = new_q
        if self.generalization is 'simple_distance':
            for s_close in self.world.get_close_states():
                self.Q[s_close, a] = new_q

            # TODO: Implement subsumption: only learn from enemies when they are on the board, and enemies take priority
