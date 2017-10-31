from abc import ABCMeta


class Agent:
    __metaclass__ = ABCMeta


class QbertAgent(Agent):
    def __init__(self, world, learner):
        self.world = world
        self.learner = learner
