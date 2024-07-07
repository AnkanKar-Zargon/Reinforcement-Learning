import numpy as np


class Agent:
    def __init__(self, env, sigma=0.1):
        self.env = env
        self.sigma = sigma
        self.position = (0, 0)
        self.theta = np.random.normal(0, 1, (25 * 4))


    def reset(self):
        self.position = (0, 0)


    def run_episode(self, policy=None):
        trajectory = [self.position]
        if policy:
            while self.position not in self.env.goal:
                action = policy[self.position]
                self.position = self.env.step(self.position, action)
                trajectory.append(self.position)
            return trajectory
        while self.position not in self.env.goal:
            action = self.env.get_action(self.position, self.sigma, self.theta)
            self.position = self.env.step(self.position, action)
            trajectory.append(self.position)
            if len(trajectory) > 1000:
                break
        return trajectory
    


    def get_trajectory(self, policy=None):
        self.reset()
        if policy:
            return self.run_episode(policy)
        return self.run_episode()


    def get_return(self, trajectory):
        return sum([self.env.get_reward(state) for _, state in enumerate(trajectory)])


    def get_average_gain(self, episode_count=20):
        return np.mean([self.get_return(self.get_trajectory()) for _ in range(episode_count)])


    def get_average_gain_per_episode(self, episode_count=75, policy=None):
        if policy != None:
            return [self.get_return(self.get_trajectory(policy)) for _ in range(episode_count)]
        return [self.get_return(self.get_trajectory()) for _ in range(episode_count)]


    


    def value_iteration(self, gamma=0.9, threshold=0.000001):
        V = np.random.rand(25)
        V[self.env.state_id[(4, 4)]] = 0
        iteration_count = 0
        while True:
            iteration_count += 1
            delta = 0
            for s in range(25):
                if s == 24:
                    continue
                state = list(self.env.state_id.keys())[s]
                v = V[s]
                V[s] = max(
                    sum(
                        [
                            p
                            * (
                                self.env.get_reward(s_prime)
                                + gamma
                                * (
                                    V[self.env.state_id[s_prime]]
                                    if s_prime in self.env.state_id
                                    else 0
                                )
                            )
                            for s_prime, p in self.env.get_possible_next_states_and_probabilities(
                                state, action
                            )
                        ]
                    )
                    for action in self.env.actions
                )
                delta = max(delta, abs(v - V[s]))
            if delta < threshold:
                break
        print(f"Value iteration converged after {iteration_count} iterations")
        return V
    


    def grad_desc(self, trial_count=300, epsilon=0.0001):
        gain_per_episode_matrix = [self.get_average_gain_per_episode()]
        max_gain = np.mean(gain_per_episode_matrix[-1])

        for trial in range(trial_count):
            cur_theta = self.theta
            std_dev_matrix = self.sigma * np.eye(*self.theta.shape)
            new_theta = np.random.multivariate_normal(self.theta, std_dev_matrix)
            self.theta = new_theta
            new_gain_per_episode = self.get_average_gain_per_episode()
            gain_per_episode_matrix.append(new_gain_per_episode)
            new_gain = np.mean(new_gain_per_episode)
            if new_gain < max_gain:
                self.theta = cur_theta
            elif new_gain > max_gain:
                max_gain = new_gain
            
        average_gains_per_trial = np.mean(gain_per_episode_matrix, axis=1)
        standard_deviation_gains_per_trial = np.std(gain_per_episode_matrix, axis=1)
        average_gains_per_episode = np.mean(gain_per_episode_matrix, axis=0)
        standard_deviation_gains_per_episode = np.std(gain_per_episode_matrix, axis=0)
        return [average_gains_per_trial, standard_deviation_gains_per_trial], [average_gains_per_episode, standard_deviation_gains_per_episode,]



    def get_optimal_policy(self, gamma=0.9):
        V_star = self.value_iteration()
        optimal_policy = {}
        for s in range(25):
            state = list(self.env.state_id.keys())[s]
            if state == (4, 4):
                continue
            best_action = state
            best_return = float("-inf")
            for action in self.env.actions:
                possible_next_states_and_probabilities = (
                    self.env.get_possible_next_states_and_probabilities(state, action)
                )
                expected_return = sum(
                    [
                        p
                        * (
                            self.env.get_reward(s_prime)
                            + gamma
                            * (
                                V_star[self.env.state_id[s_prime]]
                                if s_prime in self.env.state_id
                                else 0
                            )
                        )
                        for s_prime, p in possible_next_states_and_probabilities
                    ]
                )
                if expected_return > best_return:
                    best_return = expected_return
                    best_action = action
            optimal_policy[state] = best_action
        return optimal_policy
