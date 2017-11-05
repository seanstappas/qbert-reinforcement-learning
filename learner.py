import logging
import random
from abc import ABCMeta, abstractmethod

from actions import action_number_to_name, get_valid_action_numbers_from_state
from pickler import save_to_pickle, load_from_pickle


class Learner:
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_best_action(self, s):
        raise NotImplementedError

    @abstractmethod
    def update(self, s, a, s_next, reward):
        raise NotImplementedError


class QLearner(Learner):
    def __init__(self, world, alpha, gamma, epsilon, unexplored_threshold, unexplored_reward, exploration,
                 distance_metric, state_repr, initial_q=None, initial_n=None):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.unexplored_threshold = unexplored_threshold
        self.unexplored_reward = unexplored_reward
        self.exploration = exploration
        self.distance_metric = distance_metric
        self.Q = initial_q if initial_q is not None else {}
        self.N = initial_n if initial_n is not None else {}
        self.world = world
        self.state_repr = state_repr

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
        self.q_update(s, a, s_next, reward)

    def q_update(self, s, a, s_next, reward):
        if self.exploration is 'combined':
            self.N[s, a] = self.N.get((s, a), 0) + 1
        old_q = self.get_q(s, a)
        new_q = old_q + self.alpha * (reward + self.gamma * self.get_max_q(s_next) - old_q)
        self.Q[s, a] = new_q
        self.update_close(a, new_q)

    def get_q(self, s, a):
        return self.Q.get((s, a), 0)

    def get_best_action_random(self, s):
        actions = get_valid_action_numbers_from_state(s, self.state_repr)
        logging.debug('Valid actions: {}'.format([action_number_to_name(a) for a in actions]))
        if random.random() < self.epsilon:
            action = random.choice(actions)
            logging.debug('Randomly chose {}'.format(action_number_to_name(action)))
            return action
        random.shuffle(actions)  # If equal values, will choose random one
        max_q = float('-inf')
        max_action = None
        for a in actions:
            q = self.get_q(s, a)
            if q > max_q:
                max_q = q
                max_action = a
        return max_action

    def get_best_action_optimistic(self, s):
        actions = get_valid_action_numbers_from_state(s, self.state_repr)
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
        actions = get_valid_action_numbers_from_state(s, self.state_repr)
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

    def update_close(self, a, new_q):
        states_close, actions_close = self.world.get_close_states_actions(a, distance_metric=self.distance_metric)
        for s_close, a_close in zip(states_close, actions_close):
            self.Q[s_close, a_close] = new_q

    def save(self, filename):
        save_to_pickle(self.Q, '{}_{}'.format(filename, 'Q'))
        save_to_pickle(self.N, '{}_{}'.format(filename, 'N'))

    def load(self, filename):
        self.Q = load_from_pickle('{}_{}'.format(filename, 'Q'))
        self.N = load_from_pickle('{}_{}'.format(filename, 'N'))
        logging.debug('Loaded Q: {}'.format(self.Q))
        logging.debug('Loaded N: {}'.format(self.N))

