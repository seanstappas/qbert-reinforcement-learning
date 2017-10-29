import numpy as np

INITIAL_PARAMETERS = [
    [0],
    [0, 0],
    [0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
]  # Indicates if the desired colors are obtained at a block position


class TDParameterLearner:
    def __init__(self, alpha, gamma):
        self._theta_vector = INITIAL_PARAMETERS
        self.alpha = alpha
        self.gamma = gamma

    def parameter_update(self, desired_colors, reward):
        self._theta_vector = 0
