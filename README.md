# Reinforcement Learning
Plays the Qbert game by reinforcement learning. This is assignment 3 of the ECSE-526 class, as described [here](http://www.cim.mcgill.ca/~jer/courses/ai/assignments/as3.html). Gameplay can be seen in [this video](https://youtu.be/BKp70AEndy8).

## Installation

### Library Dependencies

The main dependency of the program is `numpy`, which can be installed via `pip`. For plotting, `matplotlib` is used.

## Usage

To run the program, use the command-line interface in `main.py`, as follows:

```
python main.py
```


To see the list of available commands, run the following:

```
python main.py --help
```

This will print the following:

```
usage: main.py [-h] [-l {info,debug,critical,warn,error}] [-e NUM_EPISODES]
               [-o LOAD_LEARNING_FILENAME] [-f SAVE_LEARNING_FILENAME]
               [-p PLOT_FILENAME] [-c CSV_FILENAME] [-d DISPLAY_SCREEN]
               [-s {simple,verbose}]
               [-a {block,enemy,friendly,subsumption,combined_verbose}]
               [-x {random,optimistic,combined}]
               [-m {manhattan,hamming,same_result}] [-r RANDOM_SEED]
               [-i SHOW_IMAGE]

Reinforcement Learning with Qbert.

optional arguments:
  -h, --help            show this help message and exit
  -l {info,debug,critical,warn,error}, --logging_level {info,debug,critical,warn,error}
                        The logging level.
  -e NUM_EPISODES, --num_episodes NUM_EPISODES
                        The number of training episodes.
  -o LOAD_LEARNING_FILENAME, --load_learning_filename LOAD_LEARNING_FILENAME
                        The pickle file to load learning data from. To run the
                        agent with pre-trained Q data, set this parameter to
                        'data'
  -f SAVE_LEARNING_FILENAME, --save_learning_filename SAVE_LEARNING_FILENAME
                        The pickle file to save learning data to.
  -p PLOT_FILENAME, --plot_filename PLOT_FILENAME
                        The filename to save a score plot to.
  -c CSV_FILENAME, --csv_filename CSV_FILENAME
                        The filename to save a score CSV file to.
  -d DISPLAY_SCREEN, --display_screen DISPLAY_SCREEN
                        Whether to display the ALE screen.
  -s {simple,verbose}, --state_representation {simple,verbose}
                        The state representation to use.
  -a {block,enemy,friendly,subsumption,combined_verbose}, --agent_type {block,enemy,friendly,subsumption,combined_verbose}
                        The agent type to use.
  -x {random,optimistic,combined}, --exploration {random,optimistic,combined}
                        The exploration mode to use.
  -m {manhattan,hamming,same_result}, --distance_metric {manhattan,hamming,same_result}
                        The distance metric to use.
  -r RANDOM_SEED, --random_seed RANDOM_SEED
                        The random seed to use.
  -i SHOW_IMAGE, --show_image SHOW_IMAGE
                        Whether to show a screenshot at the end of every
                        episode.
```

### Default Values

The default values of all the parameters can be found in the `main.py` file.


## Code Organization

The bulk of the code can be found the `actions.py`, `agent.py`, `learner.py` and `world.py` files.

Saved Q-learning values are saved to the `pickle` directory via the `pickle` python library and can be loaded via command-line arguments.

## Report

The report (`report.pdf`) and all related files (tex, plots, logs and CSV files) can be found in the `report` directory.
