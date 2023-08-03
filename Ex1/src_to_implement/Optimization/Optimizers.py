
class Sgd:

    def __init__(self, learning_rate: float):
        self.rate = learning_rate


    def calculate_update(self, weight_tensor, gradient_tensor):        #Weight update
        weight_new = weight_tensor - (self.rate * gradient_tensor)      #W'_t+1 = W'_t - n * dw
        return weight_new


