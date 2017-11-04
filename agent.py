import logging
from abc import ABCMeta, abstractmethod

from actions import action_number_to_name
from learner import QLearner
from world import QbertWorld

import matplotlib.pyplot as plt


class Agent:
    __metaclass__ = ABCMeta

    @abstractmethod
    def action(self):
        raise NotImplementedError


class QbertAgent(Agent):
    def __init__(self, agent_type='subsumption', random_seed=123, frame_skip=4, repeat_action_probability=0,
                 sound=True, display_screen=True, state_repr='adjacent', alpha=0.1, gamma=0.9, epsilon=0.5,
                 unexplored_threshold=1, unexplored_reward=50, exploration='combined', distance_metric=None):
        if agent_type is 'block':
            self.agent = QbertBlockAgent(random_seed, frame_skip, repeat_action_probability, sound, display_screen,
                                         state_repr, alpha, gamma, epsilon, unexplored_threshold, unexplored_reward,
                                         exploration, distance_metric)
        elif agent_type is 'enemy':
            self.agent = QbertEnemyAgent(random_seed, frame_skip, repeat_action_probability, sound, display_screen,
                                         state_repr, alpha, gamma, epsilon, unexplored_threshold, unexplored_reward,
                                         exploration, distance_metric)
        elif agent_type is 'friendly':
            self.agent = QbertFriendlyAgent(random_seed, frame_skip, repeat_action_probability, sound, display_screen,
                                            state_repr, alpha, gamma, epsilon, unexplored_threshold, unexplored_reward,
                                            exploration, distance_metric)
        elif agent_type is 'subsumption':
            self.agent = QbertSubsumptionAgent(random_seed, frame_skip, repeat_action_probability, sound,
                                               display_screen, state_repr, alpha, gamma, epsilon, unexplored_threshold,
                                               unexplored_reward, exploration, distance_metric)
        self.world = self.agent.world

    def action(self):
        return self.agent.action()


class QbertBlockAgent(Agent):
    def __init__(self, random_seed, frame_skip, repeat_action_probability, sound, display_screen, state_repr, alpha,
                 gamma, epsilon, unexplored_threshold, unexplored_reward, exploration, distance_metric):
        self.world = QbertWorld(random_seed, frame_skip, repeat_action_probability, sound, display_screen, state_repr)
        self.block_learner = QLearner(self.world, alpha, gamma, epsilon, unexplored_threshold, unexplored_reward,
                                      exploration, distance_metric, state_repr)

    def action(self):
        s = self.world.to_state_blocks()
        a = self.block_learner.get_best_action(s)
        block_score, friendly_score, enemy_score, enemy_penalty = self.world.perform_action(a)
        s_next = self.world.to_state_blocks()
        self.block_learner.update(s, a, s_next, block_score)
        return block_score + friendly_score + enemy_score


class QbertEnemyAgent(Agent):
    def __init__(self, random_seed, frame_skip, repeat_action_probability, sound, display_screen, state_repr, alpha,
                 gamma, epsilon, unexplored_threshold, unexplored_reward, exploration, distance_metric):
        self.world = QbertWorld(random_seed, frame_skip, repeat_action_probability, sound, display_screen, state_repr)
        self.enemy_learner = QLearner(self.world, alpha, gamma, epsilon, unexplored_threshold, unexplored_reward,
                                      exploration, distance_metric, state_repr)

    def action(self):
        s = self.world.to_state_enemies()
        a = self.enemy_learner.get_best_action(s)
        block_score, friendly_score, enemy_score, enemy_penalty = self.world.perform_action(a)
        s_next = self.world.to_state_enemies()
        self.enemy_learner.update(s, a, s_next, enemy_score + enemy_penalty)
        return block_score + friendly_score + enemy_score


class QbertFriendlyAgent(Agent):
    def __init__(self, random_seed, frame_skip, repeat_action_probability, sound, display_screen, state_repr, alpha,
                 gamma, epsilon, unexplored_threshold, unexplored_reward, exploration, distance_metric):
        self.world = QbertWorld(random_seed, frame_skip, repeat_action_probability, sound, display_screen, state_repr)
        self.friendly_learner = QLearner(self.world, alpha, gamma, epsilon, unexplored_threshold, unexplored_reward,
                                         exploration, distance_metric, state_repr)

    def action(self):
        s = self.world.to_state_friendlies()
        a = self.friendly_learner.get_best_action(s)
        block_score, friendly_score, enemy_score, enemy_penalty = self.world.perform_action(a)
        s_next = self.world.to_state_friendlies()
        self.friendly_learner.update(s, a, s_next, friendly_score)
        return block_score + friendly_score + enemy_score


class QbertSubsumptionAgent(Agent):
    def __init__(self, random_seed, frame_skip, repeat_action_probability, sound, display_screen, state_repr, alpha,
                 gamma, epsilon, unexplored_threshold, unexplored_reward, exploration, distance_metric):
        self.world = QbertWorld(random_seed, frame_skip, repeat_action_probability, sound, display_screen, state_repr)
        self.block_learner = QLearner(self.world, alpha, gamma, epsilon, unexplored_threshold, unexplored_reward,
                                      exploration, distance_metric, state_repr)
        self.friendly_learner = QLearner(self.world, alpha, gamma, epsilon, unexplored_threshold, unexplored_reward,
                                         exploration, distance_metric, state_repr)
        enemy_epsilon = 0
        self.enemy_learner = QLearner(self.world, alpha, gamma, enemy_epsilon, unexplored_threshold, unexplored_reward,
                                      exploration, distance_metric, state_repr)

        # TODO: encourage exploration heavily for block learner, less for others

    def action(self):
        enemy_present = self.world.enemy_present
        friendly_present = self.world.friendly_present
        a_enemies = None
        a_friendlies = None
        s_enemies = None
        s_friendlies = None
        if enemy_present:
            logging.debug('Enemy present!')
            s_enemies = self.world.to_state_enemies()
            a_enemies = self.enemy_learner.get_best_action(s_enemies)
        if friendly_present:
            logging.debug('Friendly present!')
            s_friendlies = self.world.to_state_friendlies()
            a_friendlies = self.friendly_learner.get_best_action(s_friendlies)
        s = self.world.to_state_blocks()
        a = self.block_learner.get_best_action(s)
        if enemy_present and a_enemies is not None:
            logging.debug('Chose enemy action!')
            chosen_action = a_enemies
        elif friendly_present and a_friendlies is not None:
            logging.debug('Chose friendly action!')
            chosen_action = a_friendlies
        else:
            logging.debug('Chose block action!')
            chosen_action = a
        if chosen_action is None:
            logging.info('None action!')
            logging.info('Current row/col : {}/{}'.format(self.world.current_row, self.world.current_col))
            logging.info('Prev block state: {}'.format(s))
            logging.info('Prev enemy state: {}'.format(s_enemies))
            logging.info('Prev friendly state: {}'.format(s_friendlies))
            plt.imshow(self.world.rgb_screen)
            plt.show()
        block_score, friendly_score, enemy_score, enemy_penalty = self.world.perform_action(chosen_action)
        if enemy_present:
            s_next_enemies = self.world.to_state_enemies()
            self.enemy_learner.update(s_enemies, chosen_action, s_next_enemies, enemy_score + enemy_penalty)
            if enemy_penalty != 0:
                logging.debug('Enemy starting state: {}'.format(s_enemies))
                logging.debug('Enemy ending state: {}'.format(s_next_enemies))
                logging.debug('Penalty: {}'.format(enemy_penalty))
            logging.debug('Enemy Q: {}'.format(self.enemy_learner.Q))
        if friendly_present:
            s_next_friendlies = self.world.to_state_friendlies()
            self.friendly_learner.update(s_friendlies, chosen_action, s_next_friendlies, friendly_score)
        s_next = self.world.to_state_blocks()
        self.block_learner.update(s, a, s_next, block_score)
        return block_score + friendly_score + enemy_score

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

        # TODO: Choose algo (SARSA, Q-learning) (Q-l better if death can occur by exploration with e-greedy: Qbert!)

        # ETTR method performed best for Qbert (minimize expected time to next positive reward)
        # ^ Aaron et al.

        # Guo: no actions can change state of game while falling from cubes (can ignore these states if possible)

        # TODO: Use pickle to save parameter weights

        # Human high scores: 15825, 27000
