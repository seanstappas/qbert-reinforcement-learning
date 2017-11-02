import logging
from abc import ABCMeta

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


class Agent:
    __metaclass__ = ABCMeta


class QbertAgent(Agent):
    def __init__(self, world, learner):
        self.world = world
        self.learner = learner

    def action(self):
        s = self.world.to_state()  # TODO: Use subsumption (3 different learners: blocks, enemies, greens...)
        a = self.learner.get_best_action(s)
        logging.debug('Chosen action: {}'.format(ACTIONS[a]))
        reward = self.world.perform_action(a)
        s_next = self.world.to_state()
        self.learner.q_update(s, a, s_next, reward, self.world.get_close_states())
        logging.debug('Q matrix: {}'.format(self.learner.Q.values()))
        logging.debug('N matrix: {}'.format(self.learner.N.values()))
        return reward
