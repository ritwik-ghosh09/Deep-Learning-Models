from .Base import BaseLayer
import numpy as np

class Flatten(BaseLayer):
    def __init__(self):
        super().__init__()


    def forward(self, inpT):        # 4D -> 2D
        self.inpT = inpT
        return np.reshape(inpT, (inpT.shape[0], -1))  #-1 placeholder lets NumPy compute width * height * depth

    def backward(self, errT):      #2D -> 4D
        width, height, depth = self.inpT.shape[1], self.inpT.shape[2], self.inpT.shape[3]
        return np.reshape(errT, (errT.shape[0], width, height, depth))