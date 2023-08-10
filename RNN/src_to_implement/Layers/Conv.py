import copy
import math
import numpy as np
import scipy.signal

from .Base import BaseLayer


class Conv(BaseLayer):
    def __init__(self, stride_shape, conv_shape, num_kernels):
        super().__init__()
        self.trainable = True

        self.num_kernels = num_kernels
        self.stride_shape = stride_shape    # int or (height, width)
        self.conv_shape = conv_shape     # kernel.shape=(channel, height, width)-> 2D output  //  (ch, ht) -> 1D output

        self.weights = np.random.uniform(0, 1, (num_kernels, *conv_shape))   #self.weight.shape = (kernels, ch, ht, wd)
        self.bias = np.random.random(num_kernels)

        self._padding = "same"
        self._optimizer = None
        self._gradient_weights = None
        self._gradient_bias = None

    def forward(self, inpT): #inpT.shape = (5 , 3, 10, 14)
        batch_size = inpT.shape[0]

        if len(inpT.shape) == 3:
            hin = inpT.shape[2]
            hout = math.ceil(hin / self.stride_shape[0])    #appends new row for decimal value
            out = np.zeros((batch_size, self.num_kernels, hout))
        if len(inpT.shape) == 4:
            hin = inpT.shape[2]
            win = inpT.shape[3]
            hout = math.ceil(hin / self.stride_shape[0])
            wout = math.ceil(win / self.stride_shape[1])
            out = np.zeros((batch_size, self.num_kernels, hout, wout))

        self.inpT = inpT  # storing for backprop

        for elem_idx in range(batch_size):
            for ker in range(self.num_kernels):
                output = scipy.signal.correlate(inpT[elem_idx], self.weights[ker],self._padding)  # same padding with stride 1 -> input size = output size
                output = output[output.shape[0] // 2]  # valid padding along channels --> drops channels -> 3D to 2D matrix(ht, wd) or 1D(ht)
                if (len(self.stride_shape) == 1):
                    output = output[::self.stride_shape[0]]  # wasteful subsampling with stride instead of pooling
                elif (len(self.stride_shape) == 2):
                    output = output[::self.stride_shape[0], ::self.stride_shape[1]]  # diff stride in diff spatial dims
                out[elem_idx, ker] = output + self.bias[ker]  #stores 2D/1D corrl. output into the output_shape corres. to 1st two dimensions
        return out

    def backward(self, errT):
        batch_size = np.shape(errT)[0]
        num_conv_kern = self.conv_shape[0]   # since #inpT channels = #conv_kernels

        weights = np.swapaxes(self.weights, 0, 1)
        weights = np.fliplr(weights)
        error_per_item = np.zeros((batch_size, self.num_kernels, *self.inpT.shape[2:]))  #(batch, ch, ht, wd)  since, #errT_ch = #forward_filters
        dX = np.zeros((batch_size, num_conv_kern, *self.inpT.shape[2:]))
        for item in range(batch_size):
            for cnv_ker in range(num_conv_kern):  #num_channels = num_conv_kernels
                if (len(self.stride_shape) == 1):
                    error_per_item[:, :, ::self.stride_shape[0]] = errT[item]   #stride over height
                else:
                    error_per_item[:, :, ::self.stride_shape[0], ::self.stride_shape[1]] = errT[item]    #stride over ht and wdt
                output = scipy.signal.convolve(error_per_item[item], weights[cnv_ker], 'same')
                output = output[output.shape[0] // 2]   #drops the dummy channels keeping the middle
                dX[item, cnv_ker] = output


        self._gradient_weights, self._gradient_bias = self.get_weights_biases_gradient(errT)
        if self.optimizer is not None:
            self.weights = copy.deepcopy(self.optimizer).calculate_update(self.weights, self._gradient_weights)
            self.bias = copy.deepcopy(self.optimizer).calculate_update(self.bias, self._gradient_bias)

        return dX

    def get_weights_biases_gradient(self, errT):    #corr
        batch_size = np.shape(errT)[0]
        num_channels = self.conv_shape[0]   #returns num_ch of inpT
        dW = np.zeros((self.num_kernels, *self.conv_shape))
        error_per_item = np.zeros((batch_size, self.num_kernels, *self.inpT.shape[2:]))
        for item in range(batch_size):
            if (len(self.stride_shape) == 1):
                error_per_item[:, :, ::self.stride_shape[0]] = errT[item]   #restores the inpT dimn
                dB = np.sum(errT, axis=(0, 2))      #dL/dB = dl/dY . 1
                padding_width = ((0, 0), (self.conv_shape[1] // 2, (self.conv_shape[1] - 1) // 2))  #half of kernel's spat_dimn
            else:               #no_padding along ch_depth but only spatial_dimn
                error_per_item[:, :, ::self.stride_shape[0], ::self.stride_shape[1]] = errT[item]
                dB = np.sum(errT, axis=(0, 2, 3))
                padding_width = ((0, 0), (self.conv_shape[1] // 2, (self.conv_shape[1] - 1) // 2),
                                 (self.conv_shape[2] // 2, (self.conv_shape[2] - 1) // 2))

            padded_X = np.pad(self.inpT[item], padding_width, mode='constant', constant_values=0)
            tmp = np.zeros((self.num_kernels, *self.conv_shape))   #shape of corr_kernels
            for ker in range(self.num_kernels):
                for ch in range(num_channels):
                    tmp[ker, ch] = scipy.signal.correlate(padded_X[ch], error_per_item[item][ker], 'valid')  #each_pad_ch * each item's channel
            dW += tmp
        return dW, dB

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @property
    def gradient_bias(self):
        return self._gradient_bias

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    def initialize(self, weights_initializer, bias_initializer):    #reinitializes the weights & biases
        self.weights = weights_initializer.initialize(self.weights.shape, np.prod(self.conv_shape),
                                                      self.num_kernels * np.prod(self.conv_shape[1:]))  #fan_out = Num_elements of weights
        self.bias = bias_initializer.initialize(self.bias.shape, 1, self.num_kernels)  #fan_out = num of kernels