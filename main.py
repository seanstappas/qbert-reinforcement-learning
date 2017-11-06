import logging

import matplotlib.pyplot as plt

from argparse import ArgumentParser

from agent import QbertAgent
from csv_utils import save_to_csv
from plotter import plot_scores

LOGGING_LEVELS = {
    'info': logging.INFO,
    'debug': logging.DEBUG,
    'warn': logging.WARN,
    'error': logging.ERROR,
    'critical': logging.CRITICAL
}


def play_learning_agent(num_episodes=2, show_image=False, load_learning_filename=None,
                        save_learning_filename=None, plot_filename=None, csv_filename=None, display_screen=False,
                        state_representation='simple', agent_type='subsumption', exploration=None,
                        distance_metric=None):
    agent = QbertAgent(display_screen=display_screen, state_representation=state_representation, agent_type=agent_type,
                       exploration=exploration, distance_metric=distance_metric)
    world = agent.world
    max_score = 0
    scores = []
    if load_learning_filename is not None:
        agent.load(load_learning_filename)
    for episode in range(num_episodes):
        total_reward = 0
        world.reset()
        while not world.ale.game_over():
            total_reward += agent.action()
        if show_image:
            plt.imshow(world.rgb_screen)
            plt.show()
        scores.append(total_reward)
        logging.info('Episode {} ended with score: {}'.format(episode + 1, total_reward))
        max_score = max(max_score, total_reward)
        world.ale.reset_game()
    if csv_filename is not None:
        save_to_csv(scores, csv_filename)
    if plot_filename is not None:
        plot_scores(scores, plot_filename)
    if save_learning_filename is not None:
        agent.save(save_learning_filename)
    logging.info('Maximum reward: {}'.format(max_score))
    logging.info('Total Q size: {}'.format(agent.q_size()))
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
    # play_learning_agent()

    # Generalization results
    distance_metric = 'no_generalization'
    play_learning_agent(num_episodes=100, plot_filename=distance_metric, csv_filename=distance_metric,
                        display_screen=False, agent_type='combined_verbose', exploration=None, distance_metric=None)

    distance_metric = 'manhattan'
    play_learning_agent(num_episodes=100, plot_filename=distance_metric, csv_filename=distance_metric,
                        display_screen=False, agent_type='combined_verbose', exploration=None,
                        distance_metric=distance_metric)

    distance_metric = 'hamming'
    play_learning_agent(num_episodes=100, plot_filename=distance_metric, csv_filename=distance_metric,
                        display_screen=False, agent_type='combined_verbose', exploration=None,
                        distance_metric=distance_metric)

    distance_metric = 'same_result'
    play_learning_agent(num_episodes=100, plot_filename=distance_metric, csv_filename=distance_metric,
                        display_screen=False, agent_type='combined_verbose', exploration=None,
                        distance_metric=distance_metric)


    # Exploration results
