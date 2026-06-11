import numpy as np

class Constant:
    # constructor receives constant value, default is 0.1
    def __init__(self, value=0.1):
        self.value = value
    def initialize(self, weights_shape, fan_in, fan_out):
        return np.full(weights_shape, self.value)
    
class UniformRandom:
     # uniform distribution in [0, 1)
    def initialize(self, weights_shape, fan_in, fan_out):
        return np.random.uniform(0, 1, weights_shape)

class Xavier:
    # zero-mean Gaussian N(0, sigma), sigma = sqrt(2 / (fan_in + fan_out))
    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2 / (fan_in + fan_out))
        return np.random.normal(0, sigma, weights_shape)

class He:
    # zero-mean Gaussian N(0, sigma), sigma = sqrt(2 / fan_in)
    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2 / fan_in)
        return np.random.normal(0, sigma, weights_shape)


