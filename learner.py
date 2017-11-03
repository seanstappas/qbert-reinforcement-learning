import logging
import random
from abc import ABCMeta, abstractmethod

from actions import action_number_to_name, get_valid_action_numbers_from_state


class Learner:
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_best_action(self, s):
        raise NotImplementedError

    @abstractmethod
    def update(self, s, a, s_next, reward):
        raise NotImplementedError


class QLearner(Learner):
    def __init__(self, world, alpha=0.5, gamma=0.9, epsilon=0.1, unexplored_threshold=5, unexplored_reward=50,
                 exploration='random', distance_metric='simple'):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.unexplored_threshold = unexplored_threshold
        self.unexplored_reward = unexplored_reward
        self.exploration = exploration
        self.distance_metric = distance_metric
        self.Q = {}
        self.N = {}
        self.world = world

    def get_best_action(self, s):
        if self.exploration is 'optimistic':
            return self.get_best_action_optimistic(s)
        elif self.exploration is 'random':
            return self.get_best_action_random(s)
        elif self.exploration is 'combined':
            return self.get_best_action_combined(s)
        else:
            return None

    def update(self, s, a, s_next, reward):
        if self.exploration is 'optimistic' or self.exploration is 'combined':
            self.q_update_optimistic(s, a, s_next, reward)
        elif self.exploration is 'random':
            self.q_update_random(s, a, s_next, reward)
        logging.debug('Q matrix: {}'.format(self.Q.values()))
        logging.debug('N matrix: {}'.format(self.N.values()))

    def get_q(self, s, a):
        return self.Q.get((s, a), 0)

    def get_best_action_random(self, s):
        actions = get_valid_action_numbers_from_state(s)
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
        actions = get_valid_action_numbers_from_state(s)
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

    def get_best_action_combined(self, s):
        actions = get_valid_action_numbers_from_state(s)
        if random.random() < self.epsilon:
            action = random.choice(actions)
            logging.debug('Randomly chose {}'.format(action_number_to_name(action)))
            return action
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
        for a in get_valid_action_numbers_from_state(s):
            max_q = max(max_q, self.Q.get((s, a), 0))
        return max_q

    def q_update_optimistic(self, s, a, s_next, reward):
        old_q = self.get_q(s, a)
        self.N[s, a] = self.N.get((s, a), 0) + 1
        new_q = old_q + self.alpha * self.N.get((s, a), 0) * (reward + self.gamma * self.get_max_q(s_next) - old_q)
        self.Q[s, a] = new_q
        self.update_close(a, new_q)

    def q_update_random(self, s, a, s_next, reward):
        old_q = self.get_q(s, a)
        new_q = old_q + self.alpha * (reward + self.gamma * self.get_max_q(s_next) - old_q)
        self.Q[s, a] = new_q
        self.update_close(a, new_q)

    def update_close(self, a, new_q):
        states_close, actions_close = self.world.get_close_states_actions(a, distance_metric=self.distance_metric)
        for s_close, a_close in zip(states_close, actions_close):
            self.Q[s_close, a_close] = new_q

            # TODO: Implement subsumption: only learn from enemies when they are on the board, and enemies take priority

