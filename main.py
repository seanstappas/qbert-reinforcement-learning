import logging
from argparse import ArgumentParser

LOGGING_LEVELS = {
    'info': logging.INFO,
    'debug': logging.DEBUG,
    'warn': logging.WARN,
    'error': logging.ERROR,
    'critical': logging.CRITICAL
}


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
    args.func(args)


if __name__ == '__main__':
    parse_command_line_arguments()
