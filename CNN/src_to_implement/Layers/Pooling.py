import numpy as np
from Layers.Base import BaseLayer


class Pooling(BaseLayer):

    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.input_shape = None
        self.windows = None

    def forward(self, input_tensor):
        self.input_shape = input_tensor.shape
        st_h, st_w = self.stride_shape  # (2,2) test case
        pool_height, pool_width = self.pooling_shape  # (2,2) test case

        batch_size, channels, height, width = input_tensor.shape
        output_height = (height - pool_height) // st_h + 1
        output_width = (width - pool_width) // st_w + 1

        windows = np.zeros((batch_size, channels, output_height, output_width, pool_height, pool_width))
        for b in range(batch_size):
            for ch in range(channels):
                for j in range(output_height):  # window sliding in y
                    for i in range(output_width):  # window sliding in x
                        h_start = j * st_h  # window starting point in y direction
                        h_end = h_start + pool_height  # window ending point in y direction
                        w_start = i * st_w  # window starting point in x direction
                        w_end = w_start + pool_width  # window ending point in x direction
                        window = input_tensor[b, ch, h_start:h_end, w_start:w_end]  # creating 2*2 arrays
                        windows[b, ch, j, i] = window  # inserting into the windows.

        self.windows = windows  # ( 2,1,2,2,2,2) shape

        output = np.zeros((batch_size, channels, output_height, output_width))  # 4D tensor.
        for b in range(batch_size):
            for ch in range(channels):
                for j in range(output_height):
                    for i in range(output_width):
                        window = windows[b, ch, j, i]
                        output[b, ch, j, i] = np.max(window)  # taking the max and returning.

        return output

    def backward(self, error_tensor):
        error_output = np.zeros(self.input_shape)
        batch_size, num_channels, output_height, output_width = error_tensor.shape  # same output shape
        st_h, st_w = self.stride_shape
        pool_height, pool_width = self.pooling_shape
        for b in range(batch_size):
            for ch in range(num_channels):
                for j in range(output_height):
                    for i in range(output_width):
                        err = error_tensor[b, ch, j, i]  # feaching the element from error_tensor
                        # windows is 6D array or this will give you a 2*2 matrix #
                        current_window = self.windows[b, ch, j, i]
                        # flatten window
                        flattened_window = current_window.flatten()  # Flatten the window to 1D
                        max_indices = np.argmax(flattened_window)  # Find the index of the maximum value

                        reshaped_window = np.zeros_like(
                            flattened_window)  # Create a zero-filled array of the same shape
                        reshaped_window[max_indices] = True  # Set the element at the index of the maximum value to 1

                        current_window = reshaped_window.reshape(current_window.shape)  # Reshape back to original shape

                        # current_window = self.pick_first_maxima(current_window)
                        error_window = err * current_window  # zero everywhere, err value otherwise

                        # pick err window
                        h_start = j * st_h
                        h_end = h_start + pool_height
                        w_start = i * st_w
                        w_end = w_start + pool_width
                        # one input can win several times in case of overlapping pooling
                        error_output[b, ch, h_start:h_end, w_start:w_end] += error_window
        return error_output
