import logging
from abc import ABCMeta

from actions import action_number_to_name


class Agent:
    __metaclass__ = ABCMeta


class QbertAgent(Agent):
    def __init__(self, world, learner):
        self.world = world
        self.learner = learner

    def action(self):
        s = self.world.to_state()  # TODO: Use subsumption (3 different learners: blocks, enemies, greens...)
        a = self.learner.get_best_action(s)
        logging.debug('Chosen action: {}'.format(action_number_to_name(a)))
        reward = self.world.perform_action(a)
        s_next = self.world.to_state()
        self.learner.update(s, a, s_next, reward)
        return reward
