import numpy as np

RAM_SIZE = 128


def initialize_parameters():
    return np.zeros(RAM_SIZE)


class TDParameterLearner:
    def __init__(self, alpha, gamma):
        self._theta_vector = initialize_parameters()
        initialize_parameters()  # Parameter for each RAM value index
        self.alpha = alpha
        self.gamma = gamma

    def parameter_update(self, ram_vector, reward):
        self._theta_vector = 0
