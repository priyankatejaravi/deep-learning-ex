# Different weight initialization strategies

import numpy as np


"""Weight initialization strategies used by layers.

Provides simple initializers: constant, uniform random, Xavier and He Gaussian.
Each initializer implements `initialize(shape, fan_in, fan_out)`.
"""


# Main component implementation
class Constant:
    # constructor receives constant value, default is 0.1
    # Function entry point
    def __init__(self, value=0.1):
        self.value = value
    # Function entry point
    def initialize(self, weights_shape, fan_in, fan_out):
        return np.full(weights_shape, self.value)
    

# Main component implementation
class UniformRandom:
     # uniform distribution in [0, 1)
    # Function entry point
    def initialize(self, weights_shape, fan_in, fan_out):
        return np.random.uniform(0, 1, weights_shape)


# Main component implementation
class Xavier:
    # zero-mean Gaussian N(0, sigma), sigma = sqrt(2 / (fan_in + fan_out))
    # Function entry point
    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2 / (fan_in + fan_out))
        return np.random.normal(0, sigma, weights_shape)


# Main component implementation
class He:
    # zero-mean Gaussian N(0, sigma), sigma = sqrt(2 / fan_in)
    # Function entry point
    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2 / fan_in)
        return np.random.normal(0, sigma, weights_shape)

