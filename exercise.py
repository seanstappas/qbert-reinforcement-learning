from __future__ import print_function

import random

ALPHA = 0.1
GAMMA = 1
EPSILON = 0.1

U = {}
R = {}
Q = {}

MAX_X = 4
MAX_Y = 3

VALID_STATES = [
    (1, 3), (2, 3), (3, 3), (4, 3),
    (1, 2), (3, 2), (4, 2),
    (1, 1), (2, 1), (3, 1), (4, 1),
]

WALL_STATE = (2, 2)

TERMINAL_POSITIVE_STATE = (4, 3)
TERMINAL_POSITIVE_REWARD = 10

TERMINAL_NEGATIVE_STATE = (4, 2)
TERMINAL_NEGATIVE_REWARD = -10

ACTIONS = {
    'N': (0, 1),
    'S': (0, -1),
    'E': (1, 0),
    'W': (-1, 0)
}

NUM_EPISODES = 1000


N = {}
N_e = 5
UNEXPLORED_REWARD = 5


def get_actions():
    return ACTIONS.keys()


def get_valid_actions_and_next_states(s):
    valid_actions = []
    valid_next_states = []
    for action in get_actions():
        s_next = result(s, action)
        if s_next in VALID_STATES:
            valid_actions.append(action)
            valid_next_states.append(s_next)
    return valid_actions, valid_next_states


def get_valid_actions(s):
    valid_actions = []
    for action in get_actions():
        s_next = result(s, action)
        if s_next in VALID_STATES:
            valid_actions.append(action)
    return valid_actions


def result(s, a):
    return s[0] + ACTIONS[a][0], s[1] + ACTIONS[a][1]


def choose_random_next_state(s):
    actions, next_states = get_valid_actions_and_next_states(s)
    return random.choice(next_states)


def initialize_rewards():
    for s in VALID_STATES:
        if s != TERMINAL_NEGATIVE_STATE and s != TERMINAL_POSITIVE_STATE:
            R[s] = 0
    R[TERMINAL_POSITIVE_STATE] = TERMINAL_POSITIVE_REWARD
    R[TERMINAL_NEGATIVE_STATE] = TERMINAL_NEGATIVE_REWARD


def initialize_utility():
    for s in VALID_STATES:
        if s != TERMINAL_NEGATIVE_STATE and s != TERMINAL_POSITIVE_STATE:
            U[s] = 0
    U[TERMINAL_POSITIVE_STATE] = TERMINAL_POSITIVE_REWARD
    U[TERMINAL_NEGATIVE_STATE] = TERMINAL_NEGATIVE_REWARD


def td_update(s, s_next):
    U[s] = U[s] + ALPHA * (R[s] + GAMMA * U[s_next] - U[s])


def terminal_state(s):
    return s == TERMINAL_POSITIVE_STATE or s == TERMINAL_NEGATIVE_STATE


def print_state(s):
    print('Current state: {]'.format(s))
    print('-----------------')
    for y in range(MAX_Y, 0, -1):
        for x in range(1, MAX_X + 1):
            char = ' '
            if (x, y) == s:
                char = 'X'
            elif (x, y) == WALL_STATE:
                char = '#'
            print('| {} '.format(char), end='')
        print()
        print('-----------------')
    print()


def print_utilities():
    print('-----------------------------------')
    for y in range(MAX_Y, 0, -1):
        for x in range(1, MAX_X + 1):
            print('| {} '.format('#####' if (x, y) == WALL_STATE else '{:5.3f}'.format(U[(x, y)])), end='')
        print()
        print('-----------------------------------')
    print()


def choose_greedy_next_state_td(s):
    actions, next_states = get_valid_actions_and_next_states(s)
    if random.random() < EPSILON:
        return random.choice(next_states)
    else:
        return max(next_states, key=U.get)  # Choose state with highest utility (assuming all have equal probability)


def initialize_q():
    for s in VALID_STATES:
        if s != TERMINAL_NEGATIVE_STATE and s != TERMINAL_POSITIVE_STATE:
            for a in get_valid_actions(s):
                Q[s, a] = 0
    Q[(3, 3), 'E'] = TERMINAL_POSITIVE_REWARD
    Q[(3, 2), 'E'] = TERMINAL_NEGATIVE_REWARD
    Q[(4, 1), 'N'] = TERMINAL_NEGATIVE_REWARD


def get_max_q(s):
    if s == TERMINAL_POSITIVE_STATE:
        return TERMINAL_POSITIVE_REWARD
    if s == TERMINAL_NEGATIVE_STATE:
        return TERMINAL_NEGATIVE_REWARD
    max_q = float('-inf')
    for a in get_valid_actions(s):
        max_q = max(max_q, Q[s, a])
    return max_q


def get_state_action_with_max_q(s, actions, next_states):
    max_q = float('-inf')
    max_action = None
    for a in actions:
        if Q[s, a] > max_q:
            max_q = Q[s, a]
            max_action = a
    return result(s, max_action), max_action


def q_update(s, a, s_next):
    Q[s, a] = Q[s, a] + ALPHA * (R[s] + GAMMA * get_max_q(s_next) - Q[s, a])


def q_update_unexplored(s, a, s_next):
    Q[s, a] = Q[s, a] + ALPHA * N.get((s, a), 0) * (R[s] + GAMMA * get_max_q(s_next) - Q[s, a])


def choose_greedy_next_state_q(s):
    actions, next_states = get_valid_actions_and_next_states(s)
    if random.random() < EPSILON:
        i = random.randint(0, len(actions) - 1)
        return next_states[i], actions[i]
    else:
        return get_state_action_with_max_q(s, actions, next_states)


def print_q():
    print('-----------------------------------')
    for y in range(MAX_Y, 0, -1):
        for x in range(1, MAX_X + 1):
            s = x, y
            q_state = '##############################'
            if s != WALL_STATE and s != TERMINAL_NEGATIVE_STATE and s != TERMINAL_POSITIVE_STATE:
                q_state = {}
                for a in get_valid_actions(s):
                    q_state[a] = '{:6.2f}'.format(Q[s, a])
            print('| {} '.format(q_state), end='')
        print()
        print('-----------------------------------')
    print()


def choose_next_state_q(s):
    actions, next_states = get_valid_actions_and_next_states(s)
    return get_state_action_with_max_q_unexplored(s, actions, next_states)


def get_state_action_with_max_q_unexplored(s, actions, next_states):
    max_q = float('-inf')
    max_action = None
    for a in actions:
        q = UNEXPLORED_REWARD if N.get((s, a), 0) < N_e else Q[s, a]
        if q > max_q:
            max_q = q
            max_action = a
    return result(s, max_action), max_action


def q1():
    """TD Learning"""
    initialize_rewards()
    initialize_utility()
    for i in range(NUM_EPISODES):
        print('Episode {}'.format(i))
        s = (1, 1)
        while not terminal_state(s):
            s_next = choose_random_next_state(s)
            td_update(s, s_next)
            s = s_next
        print_utilities()


def q2():
    """Greedy TD Learning"""
    initialize_rewards()
    initialize_utility()
    for i in range(NUM_EPISODES):
        print('Episode {}'.format(i))
        s = (1, 1)
        while not terminal_state(s):
            s_next = choose_greedy_next_state_td(s)
            td_update(s, s_next)
            s = s_next
        print_utilities()


def q4():
    """Greedy Q-learning"""
    initialize_rewards()
    initialize_q()
    for i in range(NUM_EPISODES):
        print('Episode {}'.format(i))
        s = (1, 1)
        while not terminal_state(s):
            s_next, a = choose_greedy_next_state_q(s)
            q_update(s, a, s_next)
            s = s_next
        print_q()


def q5():
    """Greedy Q-learning with favoring of unexplored states"""
    initialize_rewards()
    initialize_q()
    for i in range(NUM_EPISODES):
        print('Episode {}'.format(i))
        s = (1, 1)
        while not terminal_state(s):
            print(s)
            s_next, a = choose_next_state_q(s)
            N[s, a] = N.get((s, a), 0) + 1
            q_update(s, a, s_next)
            s = s_next
        print_q()


if __name__ == '__main__':
    # q1()
    # q2()
    # q4()
    q5()