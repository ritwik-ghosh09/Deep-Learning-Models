import numpy as np
from .Base import *

class SoftMax(BaseLayer):

    def __init__(self):
        super().__init__()
        self.y_tensor = None


    def forward(self, input_tensor):
        self.modf_tensor = input_tensor - np.max(input_tensor, axis=input_tensor.ndim-1, keepdims=True)     # x_k ---> max trick
        exp_tensor = np.exp(self.modf_tensor)
        tensor_sum = exp_tensor.sum(axis=exp_tensor.ndim-1, keepdims=True)   #sums up along axis 1,(i.e batch elements) in the 2D array with other dimn. intact
        self.y_tensor = exp_tensor / tensor_sum

        return self.y_tensor        # prediction output


    def backward(self, error_tensor):
        dx = self.y_tensor * error_tensor       # y^ * E_n
        sum = (error_tensor * self.y_tensor).sum(axis=1, keepdims=True)      # sum over j(E_nj * y^_j  )
        dx -= self.y_tensor * sum
        return dx
