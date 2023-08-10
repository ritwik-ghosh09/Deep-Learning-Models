import numpy as np


class L2_Regularizer(object):
    def __init__(self, alpha) -> None:
        self.alpha = alpha      #lambda value

    def calculate_gradient(self, weights):  #gradient
        return self.alpha*weights   #lambda * w_k

    def norm(self, weights):    #norm enhanced loss
        return self.alpha*np.sum(np.power(np.abs(weights), 2))  #lambda * ||w_k||_2

class L1_Regularizer(object):
    def __init__(self, alpha) -> None:
        self.alpha = alpha

    def calculate_gradient(self, weights):  #sub-gradient on weight
        return self.alpha*np.sign(weights)

    def norm(self, weights):
        return self.alpha*np.sum(np.abs(weights))