import logging
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
                 sound=True, display_screen=True, state_repr='adjacent_conservative', alpha=0.1, gamma=0.95,
                 epsilon=0.2, unexplored_threshold=1, unexplored_reward=100, exploration='combined',
                 distance_metric=None):
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

    def q_size(self):
        return self.agent.q_size()

    def save(self, filename):
        self.agent.save(filename)

    def load(self, filename):
        self.agent.load(filename)


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

    def q_size(self):
        return len(self.block_learner.Q)

    def save(self, filename):
        self.block_learner.save(filename)

    def load(self, filename):
        self.block_learner.load(filename)


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

    def q_size(self):
        return len(self.enemy_learner.Q)

    def save(self, filename):
        self.enemy_learner.save(filename)

    def load(self, filename):
        self.enemy_learner.load(filename)


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

    def q_size(self):
        return len(self.friendly_learner.Q)

    def save(self, filename):
        self.friendly_learner.save(filename)

    def load(self, filename):
        self.friendly_learner.load(filename)


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
        # TODO: Qbert gets stuck in corners, dies to Coily
        # TODO: Maybe problem giving green ball score to blocks? Stays at top...
        # TODO: Have number of blocks remaining in state?
        # TODO: Have distance to center in state for blocks?

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
