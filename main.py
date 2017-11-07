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
                        distance_metric=None, random_seed=123):
    logging.info('Plot filename: {}'.format(plot_filename))
    logging.info('Agent type: {}'.format(agent_type))
    logging.info('Distance metric: {}'.format(distance_metric))
    logging.info('Exploration: {}'.format(exploration))
    agent = QbertAgent(display_screen=display_screen, state_representation=state_representation, agent_type=agent_type,
                       exploration=exploration, distance_metric=distance_metric, random_seed=random_seed)
    world = agent.world
    max_score = 0
    max_level = 1
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
        max_level = max(max_level, agent.world.level)
        world.ale.reset_game()
    if csv_filename is not None:
        save_to_csv(scores, csv_filename)
    if plot_filename is not None:
        plot_scores(scores, plot_filename)
    if save_learning_filename is not None:
        agent.save(save_learning_filename)
    logging.info('Maximum reward: {}'.format(max_score))
    logging.info('Maximum level: {}'.format(max_level))
    logging.info('Total Q size: {}'.format(agent.q_size()))


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


def save_generalization_results():
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

    filename = 'subsumption_generalization'
    play_learning_agent(num_episodes=100, plot_filename=filename, csv_filename=filename,
                        display_screen=False, agent_type='subsumption', exploration=None,
                        distance_metric=None, save_learning_filename='subsumption_dangerous_no_exploration')


def save_exploration_results():
    filename = 'subsumption_random'
    play_learning_agent(num_episodes=100, plot_filename=filename, csv_filename=filename,
                        display_screen=False, agent_type='subsumption', exploration='random',
                        distance_metric=None, save_learning_filename='subsumption_dangerous_random')

    filename = 'subsumption_optimistic'
    play_learning_agent(num_episodes=100, plot_filename=filename, csv_filename=filename,
                        display_screen=False, agent_type='subsumption', exploration='optimistic',
                        distance_metric=None, save_learning_filename='subsumption_dangerous_optimistic')

    filename = 'subsumption_combined'
    play_learning_agent(num_episodes=100, plot_filename=filename, csv_filename=filename,
                        display_screen=False, agent_type='subsumption', exploration='combined',
                        distance_metric=None, save_learning_filename='subsumption_dangerous_combined')


def save_performance_results():
    filename = 'seed123'
    play_learning_agent(num_episodes=100, plot_filename=filename, csv_filename=filename,
                        display_screen=False, agent_type='subsumption', exploration='combined',
                        distance_metric=None, save_learning_filename='subsumption_dangerous_combined_123',
                        random_seed=123)

    filename = 'seed459'
    play_learning_agent(num_episodes=100, plot_filename=filename, csv_filename=filename,
                        display_screen=False, agent_type='subsumption', exploration='combined',
                        distance_metric=None, save_learning_filename='subsumption_dangerous_combined_459',
                        random_seed=459)

    filename = 'seed598'
    play_learning_agent(num_episodes=100, plot_filename=filename, csv_filename=filename,
                        display_screen=False, agent_type='subsumption', exploration='combined',
                        distance_metric=None, save_learning_filename='subsumption_dangerous_combined_598',
                        random_seed=459)


def continued_learning():
    filename = 'seed459_400'
    play_learning_agent(num_episodes=100, plot_filename=filename, csv_filename=filename,
                        display_screen=False, agent_type='subsumption', exploration='combined',
                        distance_metric=None, save_learning_filename='subsumption_dangerous_combined_459_400',
                        random_seed=459, load_learning_filename='subsumption_dangerous_combined_459_300')


def sample_play():
    play_learning_agent(num_episodes=100,
                        display_screen=True, agent_type='subsumption', exploration='combined',
                        distance_metric=None,
                        random_seed=459, load_learning_filename='subsumption_dangerous_combined_459_200')


if __name__ == '__main__':
    setup_logging('info')
    # play_learning_agent()
    # save_generalization_results()
    # save_exploration_results()
    # save_performance_results()
    # continued_learning()
    sample_play()
