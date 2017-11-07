import logging

from argparse import ArgumentParser
from agent import QbertAgent
from csv_utils import save_to_csv

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
    """
    Let the learning agent play with the specified parameters.
    """
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
            import matplotlib.pyplot as plt
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
        from plotter import plot_scores
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
    parser = ArgumentParser(description='Reinforcement Learning with Qbert.')
    parser.add_argument('-l', '--logging_level', default='info', choices=LOGGING_LEVELS.keys(),
                        help='The logging level.')
    parser.add_argument('-e', '--num_episodes', default=100, type=int, help='The number of training episodes.')
    parser.add_argument('-o', '--load_learning_filename', default=None,
                        help="The pickle file to load learning data from. To run the agent with pre-trained Q data, set"
                             " this parameter to 'data'")
    parser.add_argument('-f', '--save_learning_filename', default=None,
                        help='The pickle file to save learning data to.')
    parser.add_argument('-p', '--plot_filename', default=None,
                        help='The filename to save a score plot to.')
    parser.add_argument('-c', '--csv_filename', default=None,
                        help='The filename to save a score CSV file to.')
    parser.add_argument('-d', '--display_screen', default=False, type=bool,
                        help='Whether to display the ALE screen.')
    parser.add_argument('-s', '--state_representation', default='simple', choices=['simple', 'verbose'],
                        help='The state representation to use.')
    parser.add_argument('-a', '--agent_type', default='subsumption',
                        choices=['block', 'enemy', 'friendly', 'subsumption', 'combined_verbose'],
                        help='The agent type to use.')
    parser.add_argument('-x', '--exploration', default='combined', choices=['random', 'optimistic', 'combined'],
                        help='The exploration mode to use.')
    parser.add_argument('-m', '--distance_metric', default=None, choices=['manhattan', 'hamming', 'same_result'],
                        help='The distance metric to use.')
    parser.add_argument('-r', '--random_seed', default=123, type=int,
                        help='The random seed to use.')
    parser.add_argument('-i', '--show_image', default=False, type=bool,
                        help='Whether to show a screenshot at the end of every episode.')

    args = parser.parse_args()
    setup_logging(args.logging_level)
    play_learning_agent(num_episodes=args.num_episodes,
                        load_learning_filename=args.load_learning_filename,
                        save_learning_filename=args.save_learning_filename,
                        plot_filename=args.plot_filename,
                        csv_filename=args.csv_filename,
                        display_screen=args.display_screen,
                        state_representation=args.state_representation,
                        agent_type=args.agent_type,
                        exploration=args.exploration,
                        distance_metric=args.distance_metric,
                        random_seed=args.random_seed,
                        show_image=args.show_image)


if __name__ == '__main__':
    setup_logging('info')
    parse_command_line_arguments()
