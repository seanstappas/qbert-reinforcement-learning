import logging
from argparse import ArgumentParser
from random import randrange

from actions import action_number_to_name
from agent import QbertAgent
from learner import QLearner
from world import setup_world

LOGGING_LEVELS = {
    'info': logging.INFO,
    'debug': logging.DEBUG,
    'warn': logging.WARN,
    'error': logging.ERROR,
    'critical': logging.CRITICAL
}


def play_random_agent(world, num_episodes=10):
    for episode in range(num_episodes):
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


def play_learning_agent(world, num_episodes=10):
    learner = QLearner(world)
    agent = QbertAgent(world, learner)
    for episode in range(num_episodes):
        total_reward = 0
        world.reset_position()
        while not world.ale.game_over():
            total_reward += agent.action()
        logging.info('Episode %d ended with score: %d' % (episode, total_reward))
        world.ale.reset_game()
        # TODO: plot results here


def setup_logging(level):
    """
    Set up logging, with the specified logging level.

    :param level: the logging level
    """
    logging.basicConfig(
        format="%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(funcName)s:%(lineno)d] %(message)s",
        datefmt='%d-%m-%Y:%H:%M:%S',
        level=LOGGING_LEVELS[level])


def parse_command_line_arguments():
    """
    Parse the command-line arguments provided by the user.
    """
    parser = ArgumentParser(description='Reinforcement Learning with Q*bert.')
    parser.add_argument('-l', '--logging_level', default='info', choices=LOGGING_LEVELS.keys(),
                        help='The logging level.')

    subparsers = parser.add_subparsers()

    args = parser.parse_args()
    setup_logging(args.logging_level)
    args.func(args)


def play():
    setup_logging('info')

    world = setup_world(display_screen=True)
    play_learning_agent(world)


if __name__ == '__main__':
    play()
