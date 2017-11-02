import logging
from random import randrange

from actions import action_number_to_name
from agent import QbertAgent
from learner import QLearner
from main import setup_logging
from world import setup_world

NUM_EPISODES = 10
USE_SDL = True


# Minimal actions: ['noop', 'fire', 'up', 'right', 'left', 'down']


def play_random_agent(world):
    for episode in range(NUM_EPISODES):
        total_reward = 0
        world.reset_position()
        while not world.ale.game_over():
            legal_actions = world.valid_action_numbers()
            a = legal_actions[randrange(len(legal_actions))]
            reward = world.perform_action(a)

            logging.debug('Chosen action: {}, reward: {}'.format(action_number_to_name(a), reward))
            logging.debug('Current row/col: ({}, {}): '.format(world.current_row, world.current_col))
            logging.debug('Desired color: {}'.format(world.desired_color))
            logging.debug('Current row: {}'.format(world.current_row))
            logging.debug('Current col: {}'.format(world.current_col))
            logging.debug('Desired colors: {}'.format(world.desired_colors))
            logging.debug('Agents: {}'.format(world.agents))
            logging.debug('Reward: {}'.format(reward))

            total_reward += reward

            # plt.imshow(rgb_screen)
            # plt.show()
            # plt.savefig('report/screenshots/screenshot_{}'.format(i))
        print('Episode %d ended with score: %d' % (episode, total_reward))
        world.ale.reset_game()


def play_learning_agent(world):
    learner = QLearner(world)
    agent = QbertAgent(world, learner)
    for episode in range(NUM_EPISODES):
        total_reward = 0
        world.reset_position()
        while not world.ale.game_over():
            total_reward += agent.action()
        logging.info('Episode %d ended with score: %d' % (episode, total_reward))
        world.ale.reset_game()
        # TODO: plot results here


def play():
    world = setup_world(display_screen=True)
    # play_random_agent(world)
    play_learning_agent(world)


if __name__ == '__main__':
    setup_logging('info')
    play()

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
