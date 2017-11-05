import logging

import matplotlib.pyplot as plt

from argparse import ArgumentParser

from agent import QbertAgent

LOGGING_LEVELS = {
    'info': logging.INFO,
    'debug': logging.DEBUG,
    'warn': logging.WARN,
    'error': logging.ERROR,
    'critical': logging.CRITICAL
}


def play_learning_agent(num_episodes=1000, show_image=False):

    agent = QbertAgent()
    world = agent.world
    max_score = 0
    for episode in range(num_episodes):
        total_reward = 0
        world.reset()
        while not world.ale.game_over():
            total_reward += agent.action()
        if show_image:
            plt.imshow(world.rgb_screen)
            plt.show()
        logging.info('Episode {} ended with score: {}'.format(episode + 1, total_reward))
        max_score = max(max_score, total_reward)
        world.ale.reset_game()
    logging.info('Maximum reward: {}'.format(max_score))
    # TODO: plot results here

    # TODO: Exploration very key... getting very high scores early on because of unexplored weighting...


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


if __name__ == '__main__':
    setup_logging('info')
    play_learning_agent()
