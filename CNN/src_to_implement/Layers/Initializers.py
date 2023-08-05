import numpy as np
import math

class Constant:
    def __init__(self, val = 0.1):
        self.val = val

    def initialize(self, weights_shape, fan_in, fan_out):
        return np.full((fan_out), self.val)     #since, it's mainly used for bias

class UniformRandom:
    def initialize(self, weights_shape, fan_in, fan_out):
       return np.random.uniform(low=0, high=1, size=(weights_shape))

class Xavier:
    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = math.sqrt(2/(fan_in+fan_out))
        return np.random.normal(0, sigma, weights_shape)

class He:   #ReLU
    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = math.sqrt(2/fan_in)
        return np.random.normal(0, sigma, weights_shape)




