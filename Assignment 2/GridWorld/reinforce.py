import numpy as np
import matplotlib.pyplot as plt

class REINFORCE:
    def __init__(self, env, learning_rate=0.005, gamma=0.9):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.theta = np.random.rand(len(env.state_action_map))

    def policy(self, state, action):
        z = self.theta[self.env.state_action_map[(state, action)]]
        exp_z = np.exp(z - np.max(z))  # for numerical stability
        exp_sum = np.sum([np.exp(self.theta[self.env.state_action_map[(state, a)]] - np.max(z)) for a in range(4)])
        return exp_z / exp_sum

    def get_action(self, state):
        probabilities = [self.policy(state, a) for a in range(len(self.env.actions))]
        probabilities = np.array(probabilities)
        probabilities /= probabilities.sum()  # ensure it sums to 1
        action_index = np.random.choice(range(len(self.env.actions)), p=probabilities)
        return self.env.actions[action_index]

    def update_policy(self, rewards, states, actions):
        G = 0
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards) if np.std(rewards) > 0 else 1  # Avoid division by zero
        rewards = (rewards - mean_reward) / std_reward  # Reward normalization
        for t in reversed(range(len(rewards))):
            G = rewards[t] + self.gamma * G
            state, action = states[t], actions[t]
            log_gradient = -(G - np.sum([self.policy(state, a) * self.theta[self.env.state_action_map[(state, a)]] for a in range(4)]))
            self.theta[self.env.state_action_map[(state, action)]] += self.learning_rate * log_gradient

    def train(self, episodes=1000):
        all_rewards = []
        for episode in range(episodes):
            state = (0, 0)
            done = False
            rewards, states, actions = [], [], []
            while not done:
                action = self.get_action(state)
                next_state = self.env.step(state, action)
                reward = self.env.get_reward(next_state)
                states.append(state)
                actions.append(self.env.actions.index(action))
                rewards.append(reward)
                state = next_state
                if state in self.env.goal or state in self.env.pit:
                    done = True
            all_rewards.append(sum(rewards))
            self.update_policy(rewards, states, actions)
        return all_rewards

class BaselineREINFORCE:
    def __init__(self, env, learning_rate=0.005, gamma=0.9, beta=0.01):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.beta = beta
        self.theta = np.random.rand(len(env.state_action_map))
        self.w = np.random.rand(len(env.state_id))

    def policy(self, state, action):
        z = self.theta[self.env.state_action_map[(state, action)]]
        exp_z = np.exp(z - np.max(z))  # for numerical stability
        exp_sum = np.sum([np.exp(self.theta[self.env.state_action_map[(state, a)]] - np.max(z)) for a in range(4)])
        return exp_z / exp_sum

    def get_action(self, state):
        probabilities = [self.policy(state, a) for a in range(len(self.env.actions))]
        probabilities = np.array(probabilities)
        probabilities /= probabilities.sum()  # ensure it sums to 1
        action_index = np.random.choice(range(len(self.env.actions)), p=probabilities)
        return self.env.actions[action_index]

    def value(self, state):
        return self.w[self.env.state_id[state]]

    def update_policy(self, rewards, states, actions):
        G = 0
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards) if np.std(rewards) > 0 else 1  # Avoid division by zero
        rewards = (rewards - mean_reward) / std_reward  # Reward normalization
        for t in reversed(range(len(rewards))):
            G = rewards[t] + self.gamma * G
            state, action = states[t], actions[t]
            advantage = G - self.value(state)
            log_gradient = -(advantage - np.sum([self.policy(state, a) * self.theta[self.env.state_action_map[(state, a)]] for a in range(4)]))
            self.theta[self.env.state_action_map[(state, action)]] += self.learning_rate * log_gradient
            self.w[self.env.state_id[state]] += self.beta * advantage

    def train(self, episodes=1000):
        all_rewards = []
        for episode in range(episodes):
            state = (0, 0)
            done = False
            rewards, states, actions = [], [], []
            while not done:
                action = self.get_action(state)
                next_state = self.env.step(state, action)
                reward = self.env.get_reward(next_state)
                states.append(state)
                actions.append(self.env.actions.index(action))
                rewards.append(reward)
                state = next_state
                if state in self.env.goal or state in self.env.pit:
                    done = True
            all_rewards.append(sum(rewards))
            self.update_policy(rewards, states, actions)
        return all_rewards

