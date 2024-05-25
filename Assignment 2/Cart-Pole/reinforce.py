import numpy as np
import matplotlib.pyplot as plt


class REINFORCEAgent:
    def __init__(self, env, learning_rate=0.005, gamma=0.9):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.theta = np.random.rand(len(env.reset()), 2) * 0.01

    def policy(self, state):
        z = state.dot(self.theta)
        exp_z = np.exp(z - np.max(z))
        return exp_z / np.sum(exp_z)

    def get_action(self, state):
        prob = self.policy(state)
        return np.random.choice(range(len(prob)), p=prob)

    def update_policy(self, rewards, states, actions):
        G = 0
        policy_gradient = np.zeros_like(self.theta)
        for t in reversed(range(len(rewards))):
            G = rewards[t] + self.gamma * G
            state = states[t]
            action = actions[t]
            prob = self.policy(state)
            policy_gradient[:, action] += state * (1 - prob[action]) * G
            for a in range(len(prob)):
                if a != action:
                    policy_gradient[:, a] -= state * prob[a] * G

        self.theta += self.learning_rate * policy_gradient

    def train(self, episodes=1000):
        all_rewards = []
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            rewards, states, actions = [], [], []
            while not done:
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                state = next_state
            all_rewards.append(sum(rewards))
            self.update_policy(rewards, states, actions)
        return all_rewards

class BaselineREINFORCEAgent:
    def __init__(self, env, learning_rate=0.0001, gamma=0.1, beta=0.01):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.beta = beta
        self.theta = np.random.rand(len(env.reset()), 2) * 0.01
        self.value_weights = np.random.rand(len(env.reset())) * 0.01

    def policy(self, state):
        z = state.dot(self.theta)
        exp_z = np.exp(z - np.max(z))
        return exp_z / np.sum(exp_z)

    def get_action(self, state):
        prob = self.policy(state)
        return np.random.choice(range(len(prob)), p=prob)

    def value_function(self, state):
        return state.dot(self.value_weights)

    def update_policy(self, rewards, states, actions):
        G = 0
        policy_gradient = np.zeros_like(self.theta)
        value_gradient = np.zeros_like(self.value_weights)
        for t in reversed(range(len(rewards))):
            G = rewards[t] + self.gamma * G
            state = states[t]
            action = actions[t]
            prob = self.policy(state)
            baseline = self.value_function(state)
            advantage = G - baseline
            policy_gradient[:, action] += state * (1 - prob[action]) * advantage
            for a in range(len(prob)):
                if a != action:
                    policy_gradient[:, a] -= state * prob[a] * advantage
            value_gradient += (G - baseline) * state

        self.theta += self.learning_rate * policy_gradient
        self.value_weights += self.beta * value_gradient

    def train(self, episodes=1000):
        all_rewards = []
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            rewards, states, actions = [], [], []
            while not done:
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                state = next_state
            all_rewards.append(sum(rewards))
            self.update_policy(rewards, states, actions)
        return all_rewards