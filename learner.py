from abc import ABCMeta

NUM_ROWS = 6
NUM_COLS = 6

INITIAL_PARAMETERS = [
    [0],
    [0, 0],
    [0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
]


class Learner:
    __metaclass__ = ABCMeta


class TDParameterLearner:
    def __init__(self, alpha, gamma):
        self.desired_color_parameters = INITIAL_PARAMETERS
        self.agent_parameters = INITIAL_PARAMETERS
        self.alpha = alpha
        self.gamma = gamma

    def parameter_update(self, world, reward):
        agents = world.agents

        self.desired_color_parameters = 0
        self.agent_parameters = 0

    def utility(self, world):
        utility = 0
        # Desired color utility
        for row in range(NUM_ROWS):
            for col in range(row + 1):
                utility += self.desired_color_parameters[row][col] * world.desired_colors[row][col]

        # Agent utility
        for row in range(NUM_ROWS):
            for col in range(row + 1):
                utility += self.agent_parameters[row][col] * world.agents[row][col]

        return utility
