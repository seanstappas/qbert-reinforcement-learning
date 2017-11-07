from main import play_learning_agent


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
    filename = 'seed459_2200'
    play_learning_agent(num_episodes=1600, plot_filename=filename, csv_filename=filename,
                        display_screen=False, agent_type='subsumption', exploration='combined',
                        distance_metric=None, save_learning_filename='subsumption_dangerous_combined_459_2200',
                        random_seed=459, load_learning_filename='subsumption_dangerous_combined_459_600')


def sample_play():
    play_learning_agent(num_episodes=100,
                        display_screen=True, agent_type='subsumption', exploration='combined',
                        distance_metric=None,
                        random_seed=459, load_learning_filename='subsumption_dangerous_combined_459_400')


if __name__ == '__main__':
    # play_learning_agent()
    # save_generalization_results()
    # save_exploration_results()
    # save_performance_results()
    continued_learning()
    # sample_play()
