import numpy as np

INITIAL_PARAMETERS = [
    [0],
    [0, 0],
    [0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
]


class TDParameterLearner:
    def __init__(self, alpha, gamma):
        self.desired_color_parameters = INITIAL_PARAMETERS
        self.agent_parameters = INITIAL_PARAMETERS
        self.alpha = alpha
        self.gamma = gamma

    def parameter_update(self, world, reward):
        self.desired_color_parameters = 0
        self.agent_parameters = 0
