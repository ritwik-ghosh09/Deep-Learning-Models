from .Base import *
import numpy as np

class FullyConnected(BaseLayer):

    def __init__(self, input_size, output_size):

        self.input_size = input_size
        self.output_size = output_size
        super().__init__()
        self.trainable = True       #trainable parameter w
        self.optimizer = None
        self.weights = np.random.uniform(low=0, high=1, size=(self.input_size+1, self.output_size))  #since, output dimn is inp_size x output_siz
                                                                                                    # adds an extra row for bias in W'

    def forward(self, input_tensor):

        rows = input_tensor.shape[0]
        ones_matrix = np.ones((rows, 1))
        self.modf_tensor = np.concatenate((input_tensor, ones_matrix), axis=1)   #adds the ones_matrix

        self.output_tensor = np.dot(self.modf_tensor, self.weights)     # Y^' = X' * W'

        return self.output_tensor


    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, val):
        self._optimizer = val



    def backward(self, error_tensor):

        self.prev_err_tensor = np.dot(error_tensor, (self.weights[:self.input_size]).T)     # E'_n-1 = E'_n * W'.T

        self.gradient_weights = np.dot((self.modf_tensor).T, error_tensor)      #gradient_tensor dw = X'.T * E'_n

        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)

        return self.prev_err_tensor

    '''@property
    def gradient_weights(self):
        return self.gradient_wt
    
    @gradient_weights.setter
    def gradient_weights(self, value):
        self.gradient_wt = value'''








