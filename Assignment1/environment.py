import numpy as np
from itertools import product


class GridWorld:
    def __init__(self):
        self.state_id = {
            state: id for id, state in enumerate(list(product(range(5), range(5))))
        }
        self.actions = [(-1, 0), (0, 1), (0, -1), (1, 0)]
        
        self.state_action_map = {
            (state, action): i
            for i, (state, action) in enumerate( product(list(self.state_id.keys()), [0, 1, 2, 3]))
        }

        self.unintended_actions = { (-1, 0): [(0, 1), (0, -1)], (0, 1): [(1, 0), (-1, 0)], (0, -1): [(-1, 0), (1, 0)], (1, 0): [(0, -1), (0, 1)],}

        self.obstacles = [(2, 2), (3, 2)]
        self.pit = [(4, 2)]
        self.goal = [(4, 4)]
        self.discount = 0.9
        self.transition_prob = {
            "intended": 0.8,
            "cw": 0.05,
            "ccw": 0.05,
            "stay": 0.1,
        }
        self.reward = {"goal": 10, "pit": -10, "step": 0}

    

    def get_possible_next_states_and_probabilities(self, state, action):
        if state in self.obstacles:
            return [[state, 0]]
        intended = action
        a = self.unintended_actions[intended][0]
        ab = self.unintended_actions[intended][1]
        intended_state = (state[0] + intended[0], state[1] + intended[1])
        a_state = (state[0] + a[0], state[1] + a[1])
        ab_state = (state[0] + ab[0], state[1] + ab[1])
        stay_state = state
        intended_prob = self.transition_prob["intended"]
        a_prob = self.transition_prob["cw"]
        ab_prob = self.transition_prob["ccw"]
        stay_prob = self.transition_prob["stay"]

        if intended_state in self.obstacles:
            intended_prob = 0
        if a_state in self.obstacles:
            a_prob = 0
        if ab_state in self.obstacles:
            ab_prob = 0
        check_if_in_grid = lambda x: 0 <= x[0] < 5 and 0 <= x[1] < 5
        if not check_if_in_grid(intended_state):
            intended_prob = 0
        if not check_if_in_grid(a_state):
            a_prob = 0
        if not check_if_in_grid(ab_state):
            ab_prob = 0
        return [[intended_state, intended_prob], [a_state, a_prob], [ab_state, ab_prob], [stay_state, stay_prob],
        ]
    

    def get_transition_probability_map(self, state, action):
        intended = action
        a = self.unintended_actions[intended][0] 
        ab = self.unintended_actions[intended][1]
        intended_state = (state[0] + intended[0], state[1] + intended[1])
        a_state = (state[0] + a[0], state[1] + a[1])
        ab_state = (state[0] + ab[0], state[1] + ab[1])
        stay_state = state

        if intended_state in self.obstacles:
            intended_state = state
        if a_state in self.obstacles:
            a_state = state
        if ab_state in self.obstacles:
            ab_state = state
        check_if_in_grid = lambda x: 0 <= x[0] < 5 and 0 <= x[1] < 5
        if not check_if_in_grid(intended_state):
            intended_state = state
        if not check_if_in_grid(a_state):
            a_state = state
        if not check_if_in_grid(ab_state):
            ab_state = state
        return [ [intended_state, self.transition_prob["intended"]], [a_state, self.transition_prob["cw"]], [ab_state, self.transition_prob["ccw"]], [stay_state, self.transition_prob["stay"]],]

    def get_reward(self, state):
        if isinstance(state, int):
            for k, v in self.state_id.items():
                if v == state:
                    state = k
                    break
        if state in self.goal:
            reward = self.reward["goal"]
        elif state in self.pit:
            reward = self.reward["pit"]
        else:
            reward = self.reward["step"]
        return reward

    def get_policy(self, state, action, sigma, theta):
        return np.exp(sigma * theta[self.state_action_map[(state, action)]]) / np.sum([np.exp(sigma * theta[self.state_action_map[(state, a)]]) for a in range(4)])
    


    def get_action(self, state, sigma, theta):
        probabilities = [self.get_policy(state, a, sigma, theta) for a in range(len(self.actions))]
        action_index = np.random.choice(range(len(self.actions)), p=probabilities)
        return self.actions[action_index]
    


    def step_helper(self, state, action):
        transition_prob_map = self.get_transition_probability_map(state, action)
        possible_next_states = [x for [x, _] in transition_prob_map]
        probabilities = [y for [_, y] in transition_prob_map]
        next_state_index = np.random.choice(range(len(possible_next_states)), p=probabilities)
        next_state = possible_next_states[next_state_index]
        return next_state
    

    def step(self, state, action):
        next_state = self.step_helper(state, action)
        return next_state
