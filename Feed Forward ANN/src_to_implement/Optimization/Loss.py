import numpy as np

class CrossEntropyLoss():
    def __init__(self):
        self.pred = None

    def forward(self, pred_tensor, label_tensor):   #tensors are 2D with every row = every element of batch

        self.pred = pred_tensor
        loss = label_tensor * np.log(pred_tensor + np.finfo(float).eps)     #p_k * log(q_k) --> returns only the values for y = 1
        return -np.sum(loss)    #sums up the loss of every sample element



    def backward(self, label_tensor):
        error_tensor = -(label_tensor / (self.pred + np.finfo(float).eps))      # -y / (y^ + eps)
        return error_tensor