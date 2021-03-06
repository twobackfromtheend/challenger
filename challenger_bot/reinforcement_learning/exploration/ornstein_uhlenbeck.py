import numpy as np


class OrnsteinUhlenbeck:
    """
    See  https://github.com/keras-rl/keras-rl/blob/master/rl/random.py
    """

    def __init__(self, theta, mu: float = 0, sigma: float = 1, dt=1e-2, size: int = 1):
        self.mu = mu
        self.sigma = sigma
        self.size = size
        self.theta = theta
        self.dt = dt
        self.previous_noise = None
        self.reset_states()

    def get_action(self, action: np.ndarray):
        action = action + self.sample()
        return action

    def sample(self):
        noise = self.previous_noise + \
                self.theta * (self.mu - self.previous_noise) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)
        self.previous_noise = noise
        return noise

    def reset_states(self):
        self.previous_noise = np.random.normal(self.mu, self.sigma, size=self.size)


if __name__ == '__main__':
    size = 5
    ou = OrnsteinUhlenbeck(theta=0.15, sigma=0.05, size=size)
    for i in range(100):
        action = ou.get_action(np.zeros(size))
        print(action)

