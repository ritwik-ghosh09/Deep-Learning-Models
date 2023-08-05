'''from .Base import BaseLayer
from .Initializers import *

class Conv(BaseLayer):
    def __init__(self, stride_shape, conv_shape, num_kernels):
        super().__init__()
        self.trainable = True
        self.str_shp = stride_shape     # int or (height, width)
        self.conv_shape = conv_shape    # kernel.shape(channel, height, width)-> 2D output
        self.num_kernels = num_kernels
        self.weights = np.random.uniform(low=0, high=1, size=(num_kernels, *conv_shape))   #initializes the weights of Kernel filter with uniform randomness
        self.bias = np.random.random(self.num_kernels)      # sets the bias for every kernel filter

    def forward(self, inpT):    #inpT.shape = (batch, channel, height, width) or (batch, channel, height)
        padded_height = inpT.shape[2] + 2 * (self.conv_shape[1] // 2)  # zero-padding as per kernel height
        padded_width = inpT.shape[3] + 2*(self.conv_shape[2] // 2)

        if len(inpT.shape) > 3:   #handles 2D output
               #stride is a tuple
            output_shape = (inpT.shape[0], self.num_kernels, ((padded_height-self.conv_shape[1]) / self.str_shp[0]) + 1, ((padded_width-self.conv_shape[2]) / self.str_shp[1]) +1 )
        else:   # inpT.shape = (batch, channel, height) -> 1D output
            paddedW = 1 + 2(self.conv_shape[2] // 2)
            if len(self.str_shp) > 1:
                output_shape = (inpT.shape[0], self.num_kernels, ((padded_height- self.conv_shape[1]) / self.str_shp[0]) +1, ((paddedW - self.conv_shape[2]) / self.str_shp[1]) + 1)
            else:
                output_shape = (inpT.shape[0], self.num_kernels, ((padded_height- self.conv_shape[1]) / self.str_shp[0]) +1, ((paddedW - self.conv_shape[2]) / self.str_shp[0]) + 1)

        print(f"inputTensor shape, Tensor: {inpT.shape}, {inpT}")
        #print(f"bias_shape, bias_generated: {self.bias.shape}, {self.bias}")
        #print(f"weight_shape, weight_generated {self.weights.shape}, {self.weights}")

        return 2
'''

import numpy as np

x = np.arange(120).reshape(2, 3, 4, 5)
y = x[:, :, ::2, ::2]
'''print(f"x is {x}")
print("_" *20)
print(y)
print(y.shape)'''

import numpy as np

# Assuming you have a 2D matrix named 'matrix'
# Create an example matrix
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# Perform padding using np.pad()
padded_matrix = np.pad(matrix, ((5, 1), (1, 0)), mode='constant', constant_values=0)

# Print the padded matrix
print(padded_matrix)


