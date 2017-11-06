import logging
import random
from abc import ABCMeta, abstractmethod

from learner import QLearner
from world import QbertWorld


class Agent:
    __metaclass__ = ABCMeta

    @abstractmethod
    def action(self):
        raise NotImplementedError


class QbertAgent(Agent):
    def __init__(self, agent_type='subsumption', random_seed=123, frame_skip=4, repeat_action_probability=0,
                 sound=True, display_screen=True, alpha=0.1, gamma=0.95,
                 epsilon=0.2, unexplored_threshold=1, unexplored_reward=100, exploration='combined',
                 distance_metric=None, combined_reward=True, state_representation='simple'):
        if agent_type is 'block':
            self.agent = QbertBlockAgent(random_seed, frame_skip, repeat_action_probability, sound, display_screen,
                                         alpha, gamma, epsilon, unexplored_threshold, unexplored_reward,
                                         exploration, distance_metric, state_representation)
        elif agent_type is 'enemy':
            self.agent = QbertEnemyAgent(random_seed, frame_skip, repeat_action_probability, sound, display_screen,
                                         alpha, gamma, epsilon, unexplored_threshold, unexplored_reward,
                                         exploration, distance_metric, state_representation)
        elif agent_type is 'friendly':
            self.agent = QbertFriendlyAgent(random_seed, frame_skip, repeat_action_probability, sound, display_screen,
                                            alpha, gamma, epsilon, unexplored_threshold, unexplored_reward,
                                            exploration, distance_metric, state_representation)
        elif agent_type is 'subsumption':
            self.agent = QbertSubsumptionAgent(random_seed, frame_skip, repeat_action_probability, sound,
                                               display_screen, alpha, gamma, epsilon, unexplored_threshold,
                                               unexplored_reward, exploration, distance_metric, combined_reward,
                                               state_representation)
        elif agent_type is 'combined_verbose':
            self.agent = QbertCombinedVerboseAgent(random_seed, frame_skip, repeat_action_probability, sound,
                                                   display_screen, alpha, gamma, epsilon, unexplored_threshold,
                                                   unexplored_reward, exploration, distance_metric)

        self.world = self.agent.world

    def action(self):
        return self.agent.action()

    def q_size(self):
        return self.agent.q_size()

    def save(self, filename):
        self.agent.save(filename)

    def load(self, filename):
        self.agent.load(filename)


class QbertBlockAgent(Agent):
    def __init__(self, random_seed, frame_skip, repeat_action_probability, sound, display_screen, alpha,
                 gamma, epsilon, unexplored_threshold, unexplored_reward, exploration, distance_metric,
                 state_representation):
        if state_representation is 'simple':
            state_repr = 'along_direction'
        else:
            state_repr = 'verbose'
        self.world = QbertWorld(random_seed, frame_skip, repeat_action_probability, sound, display_screen,
                                block_state_repr=state_repr)
        self.block_learner = QLearner(self.world, alpha, gamma, epsilon, unexplored_threshold, unexplored_reward,
                                      exploration, distance_metric, state_repr)

    def action(self):
        s = self.world.to_state_blocks()
        a = self.block_learner.get_best_action(s)
        block_score, friendly_score, enemy_score, enemy_penalty = self.world.perform_action(a)
        s_next = self.world.to_state_blocks()
        self.block_learner.update(s, a, s_next, block_score)
        return block_score + friendly_score + enemy_score

    def q_size(self):
        return len(self.block_learner.Q)

    def save(self, filename):
        self.block_learner.save(filename)

    def load(self, filename):
        self.block_learner.load(filename)


class QbertEnemyAgent(Agent):
    def __init__(self, random_seed, frame_skip, repeat_action_probability, sound, display_screen, alpha,
                 gamma, epsilon, unexplored_threshold, unexplored_reward, exploration, distance_metric,
                 state_representation):
        if state_representation is 'simple':
            state_repr = 'adjacent_conservative'
        else:
            state_repr = 'verbose'
        self.world = QbertWorld(random_seed, frame_skip, repeat_action_probability, sound, display_screen,
                                enemy_state_repr=state_repr)
        self.enemy_learner = QLearner(self.world, alpha, gamma, epsilon, unexplored_threshold, unexplored_reward,
                                      exploration, distance_metric, state_repr)

    def action(self):
        s = self.world.to_state_enemies()
        a = self.enemy_learner.get_best_action(s)
        block_score, friendly_score, enemy_score, enemy_penalty = self.world.perform_action(a)
        s_next = self.world.to_state_enemies()
        self.enemy_learner.update(s, a, s_next, enemy_score + enemy_penalty)
        return block_score + friendly_score + enemy_score

    def q_size(self):
        return len(self.enemy_learner.Q)

    def save(self, filename):
        self.enemy_learner.save(filename)

    def load(self, filename):
        self.enemy_learner.load(filename)


class QbertFriendlyAgent(Agent):
    def __init__(self, random_seed, frame_skip, repeat_action_probability, sound, display_screen, alpha,
                 gamma, epsilon, unexplored_threshold, unexplored_reward, exploration, distance_metric,
                 state_representation):
        if state_representation is 'simple':
            state_repr = 'simple'
        else:
            state_repr = 'verbose'
        self.world = QbertWorld(random_seed, frame_skip, repeat_action_probability, sound, display_screen,
                                friendly_state_repr=state_repr)
        self.friendly_learner = QLearner(self.world, alpha, gamma, epsilon, unexplored_threshold, unexplored_reward,
                                         exploration, distance_metric, state_repr)

    def action(self):
        s = self.world.to_state_friendlies()
        a = self.friendly_learner.get_best_action(s)
        block_score, friendly_score, enemy_score, enemy_penalty = self.world.perform_action(a)
        s_next = self.world.to_state_friendlies()
        self.friendly_learner.update(s, a, s_next, friendly_score)
        return block_score + friendly_score + enemy_score

    def q_size(self):
        return len(self.friendly_learner.Q)

    def save(self, filename):
        self.friendly_learner.save(filename)

    def load(self, filename):
        self.friendly_learner.load(filename)


class QbertCombinedVerboseAgent(Agent):
    def __init__(self, random_seed, frame_skip, repeat_action_probability, sound, display_screen, alpha,
                 gamma, epsilon, unexplored_threshold, unexplored_reward, exploration, distance_metric):
        state_repr = 'verbose'
        self.world = QbertWorld(random_seed, frame_skip, repeat_action_probability, sound, display_screen)
        self.learner = QLearner(self.world, alpha, gamma, epsilon, unexplored_threshold, unexplored_reward,
                                exploration, distance_metric, state_repr)

    def action(self):
        s = self.world.to_state_combined_verbose()
        a = self.learner.get_best_action(s)
        block_score, friendly_score, enemy_score, enemy_penalty = self.world.perform_action(a)
        s_next = self.world.to_state_combined_verbose()
        self.learner.update(s, a, s_next, block_score + friendly_score + enemy_score + enemy_penalty)
        return block_score + friendly_score + enemy_score

    def q_size(self):
        return len(self.learner.Q)

    def save(self, filename):
        self.learner.save(filename)

    def load(self, filename):
        self.learner.load(filename)


class QbertSubsumptionAgent(Agent):
    def __init__(self, random_seed, frame_skip, repeat_action_probability, sound, display_screen, alpha,
                 gamma, epsilon, unexplored_threshold, unexplored_reward, exploration, distance_metric,
                 combined_reward, state_representation):
        if state_representation is 'simple':
            state_repr = 'simple'
        else:
            state_repr = 'verbose'
        block_state_repr = 'adjacent'
        enemy_state_repr = 'adjacent_conservative'
        friendly_state_repr = 'simple'
        self.world = QbertWorld(random_seed, frame_skip, repeat_action_probability, sound, display_screen,
                                block_state_repr=block_state_repr,
                                enemy_state_repr=enemy_state_repr,
                                friendly_state_repr=friendly_state_repr)
        self.block_learner = QLearner(self.world, alpha, gamma, epsilon, unexplored_threshold, unexplored_reward,
                                      exploration, distance_metric, state_repr=block_state_repr, tag='blocks')
        self.friendly_learner = QLearner(self.world, alpha, gamma, epsilon, unexplored_threshold, unexplored_reward,
                                         exploration, distance_metric, state_repr=friendly_state_repr,
                                         tag='friendlies')
        enemy_epsilon = 0
        self.enemy_learner = QLearner(self.world, alpha, gamma, enemy_epsilon, unexplored_threshold, unexplored_reward,
                                      exploration, distance_metric, state_repr=enemy_state_repr, tag='enemies')
        self.combined_reward = combined_reward

        # TODO: encourage exploration heavily for block learner, less for others
        # TODO: Qbert gets stuck in corners, dies to Coily
        # TODO: Have number of blocks remaining in state?
        # TODO: Have distance to center in state for blocks?
        # TODO: Have level in state? To encourage exploration at every level?
        # TODO: Level-up learner?

    def action(self):
        enemy_present = self.world.is_enemy_nearby()
        friendly_present = self.world.is_friendly_nearby()
        a_enemies = None
        a_friendlies = None
        s_enemies = None
        s_friendlies = None
        if enemy_present:
            logging.debug('Enemy present!')
            s_enemies = self.world.to_state_enemies()
            a_enemies = self.enemy_learner.get_best_actions(s_enemies)
        if friendly_present:
            logging.debug('Friendly present!')
            s_friendlies = self.world.to_state_friendlies()
            a_friendlies = self.friendly_learner.get_best_actions(s_friendlies)
        s = self.world.to_state_blocks()
        a = self.block_learner.get_best_actions(s)
        if enemy_present and len(a_enemies) > 0:
            logging.debug('Chose enemy action!')
            if len(a_enemies) > 1:
                logging.debug('Broke tie!')
                chosen_action = self.block_learner.get_best_action(s, a_enemies)
            else:
                chosen_action = a_enemies[0]
        elif friendly_present and len(a_friendlies) > 0:
            logging.debug('Chose friendly action!')
            chosen_action = a_friendlies[0]
        else:
            logging.debug('Chose block action!')
            chosen_action = random.choice(a)
        block_score, friendly_score, enemy_score, enemy_penalty = self.world.perform_action(chosen_action)
        if enemy_present:
            s_next_enemies = self.world.to_state_enemies()
            self.enemy_learner.update(s_enemies, chosen_action, s_next_enemies, enemy_score + enemy_penalty)
        if friendly_present:
            s_next_friendlies = self.world.to_state_friendlies()
            self.friendly_learner.update(s_friendlies, chosen_action, s_next_friendlies, friendly_score)
        s_next = self.world.to_state_blocks()
        combined_score = block_score if self.combined_reward else block_score
        self.block_learner.update(s, chosen_action, s_next, combined_score)
        return block_score + friendly_score + enemy_score

    def q_size(self):
        return len(self.block_learner.Q) + \
               len(self.friendly_learner.Q) + \
               len(self.enemy_learner.Q)

    def save(self, filename):
        self.block_learner.save('{}_{}'.format(filename, 'block'))
        self.friendly_learner.save('{}_{}'.format(filename, 'friendly'))
        self.enemy_learner.save('{}_{}'.format(filename, 'enemy'))

    def load(self, filename):
        self.block_learner.load('{}_{}'.format(filename, 'block'))
        self.friendly_learner.load('{}_{}'.format(filename, 'friendly'))
        self.enemy_learner.load('{}_{}'.format(filename, 'enemy'))
        # Human high scores: 15825, 27000
