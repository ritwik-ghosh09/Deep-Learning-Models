from .Base import *

class ReLU(BaseLayer):

    def __init__(self):
        super().__init__()


    def forward(self, input_tensor):

        input_tensor[input_tensor <= 0] = 0     #sets zeros and negatives to zero for filtering out those corr. neurons
        self.output_tensor = input_tensor
        return self.output_tensor


    def backward(self, error_tensor):
        self.output_tensor[self.output_tensor > 0] = 1     # sets the dy/dx matrix by changing gradients of all positives to 1 in output_tensor
        return error_tensor * self.output_tensor        # (dL/dy). (dy/dx)
