# Code source: https://github.com/wpiszlogin/driver_critic/blob/main/tools.py

import numpy as np

class NoiseGenerator:
    def __init__(self, mean, std_dev, theta=0.3, dt=5e-2):
        self.theta = theta
        self.dt = dt
        self.mean = mean
        self.std_dev = std_dev

        if mean.shape != std_dev.shape:
            raise ValueError('Mean shape: {} and std_dev shape: {} should be the same!'.format(
                mean.shape, std_dev.shape))

        # This shape will be generated
        self.x_shape = mean.shape
        self.x = None

        self.reset()

    def reset(self):
        # Reinitialize generator
        self.x = np.zeros_like(self.x_shape)

    def generate(self, std_dev_factor=None):
        if std_dev_factor is not None:
            self.std_dev[0] = std_dev_factor
            self.std_dev[1] = std_dev_factor
            self.std_dev[2] = std_dev_factor

        # The result is based on the old value
        # The second segment will keep values near a mean value
        # It uses normal distribution multiplied by a standard deviation
        self.x = (self.x
                  + self.theta * (self.mean - self.x) * self.dt
                  + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.x_shape))
        
        # TODO: check if decaying noise helps
        # self.std_dev = self.std_dev * 0.9999
        return self.x
