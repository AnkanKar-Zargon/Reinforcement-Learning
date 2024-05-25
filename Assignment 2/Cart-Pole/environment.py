import numpy as np
import matplotlib.pyplot as plt

class CartPoleEnv:
    def __init__(self):
        self.gravity = 9.8
        self.mass_cart = 1.0
        self.mass_pole = 0.1
        self.total_mass = self.mass_cart + self.mass_pole
        self.length = 0.5  # half the pole's length
        self.polemass_length = self.mass_pole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # time interval for each step

        self.x_threshold = 3.0
        self.theta_threshold_radians = 5 * np.pi / 12
        self.reset()

    def reset(self):
        self.state = np.zeros(4)
        self.steps_beyond_done = None
        return self.state

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.mass_pole * costheta ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x = x + self.tau * x_dot
        x_dot = np.clip(x_dot + self.tau * xacc, -10, 10)
        theta = theta + self.tau * theta_dot
        theta_dot = np.clip(theta_dot + self.tau * thetaacc, -np.pi, np.pi)

        self.state = (x, x_dot, theta, theta_dot)

        done = x < -self.x_threshold or x > self.x_threshold or theta < -self.theta_threshold_radians or theta > self.theta_threshold_radians

        if done:
            reward = 0.0
        else:
            reward = 1.0

        return np.array(self.state), reward, done, {}

    def render(self):
        pass  # we are not rendering
