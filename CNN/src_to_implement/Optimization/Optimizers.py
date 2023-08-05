import numpy as np
class Sgd:
    def __init__(self, learning_rate: float):
        self.rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        sgd_weight = weight_tensor - (self.rate * gradient_tensor)      #W'_t+1 = W'_t - n * dw
        return sgd_weight

class SgdWithMomentum:
    def __init__(self, learn_rate, momn_rate):
        self.learning_rate = learn_rate
        self.momentum_rate = momn_rate
        self.v_k = 0
    def calculate_update(self, weightT, gradT):
        self.v_k = self.momentum_rate * self.v_k - (self.learning_rate * gradT)  # v_k-1 = 0 with no history at the beginning
        new_weight = weightT + self.v_k
        return new_weight


class Adam:
    def __init__(self, learn_rate, mu, rho):
        self.learn_rate = learn_rate
        self.mu = mu
        self.rho = rho
        self.k = 1
        self.v_k = 0
        self.r_k = 0

    def calculate_update(self, weightT, gradT):

        self.v_k = (self.mu * self.v_k) + (1 - self.mu) * gradT     # v_k_1 = v_k-1
        self.r_k = (self.rho * self.r_k) + (1 - self.rho) * gradT * gradT

        v_corr = self.v_k / (1- np.power(self.mu, self.k))
        r_corr = self.r_k / (1 - np.power(self.rho, self.k))
        self.k += 1
        return weightT - (self.learn_rate * v_corr / (np.sqrt(r_corr) + np.finfo(float).eps))
