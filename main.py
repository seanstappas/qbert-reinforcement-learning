import logging
from argparse import ArgumentParser

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


def play_learning_agent(world, num_episodes=10, exploration='random', generalization='simple_distance'):
    learner = QLearner(world, exploration=exploration, generalization=generalization)
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
    play_learning_agent(world, num_episodes=20, exploration='optimistic', generalization='simple_distance')


if __name__ == '__main__':
    play()
