import logging
from abc import ABCMeta, abstractmethod

from actions import action_number_to_name


class Agent:
    __metaclass__ = ABCMeta

    @abstractmethod
    def action(self):
        raise NotImplementedError


class QbertAgent(Agent):
    def __init__(self, world, learner):
        self.world = world
        self.learner = learner

    def action(self):
        s = self.world.to_state()  # TODO: Use subsumption (3 different learners: blocks, enemies, greens...)
        logging.debug('Current state: {}'.format(s))
        a = self.learner.get_best_action(s)
        logging.debug('Chosen action: {}'.format(action_number_to_name(a)))
        reward = self.world.perform_action(a)
        s_next = self.world.to_state()
        self.learner.update(s, a, s_next, reward)
        return reward

    # TODO: Penalty for losing life? (equivalent to hitting an enemy... only valid for enemy subsystem?)

    # TODO: see and select actions on every kth frame: recommended every 4th frame

    # TODO: OR act at 12 steps/second (frame skip=5 within the stellarc configuration file.)

    # TODO: construct feature set (Basic or RAM best for Qbert, as shown in Bellemare et al.)
    # Aaron et al.: Tile coding is the most practical feature extraction technique. We also experimented
    # with convolutional features, where a set of predefined filters were run over the image each
    # step. The large number of convolutions required was too slow, at least using OpenCV2 or
    # Theano3 convolutional codes.
    # We performed our experiments using a variant of the BASIC representation, limited
    # to the SECAM color set. This representation is simply an encoding of the screen with a
    # courser grid, with a resolution of 14x16. Colors that occur in each 15x10 block are encoded
    # using indicator features, 1 for each of the 8 SECAM colors. Background subtraction is used
    # before encoding, as detailed in Bellemare et al. (2013).

    # TODO: Choose algorithm (SARSA, Q-learning) (Q-l. better if death can occur by exploration with e-greedy: Qbert!)
    # ETTR method performed best for Qbert (minimize expected time to next positive reward)
    # ^ Aaron et al.

    # Guo: no actions can change state of game while falling from cubes (can ignore these states if possible)

    # Probable best choice: Basic + Q-learning

    # TODO: Use pickle to save parameter weights

    # TODO: Only consider left, right, up, down actions

    # Human high scores: 15825, 27000
