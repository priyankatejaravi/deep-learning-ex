import numpy as np
from Layers.Base import BaseLayer


class Pooling(BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.max_positions = None
        self.input_shape = None

    def forward(self, input_tensor):
        self.input_shape = input_tensor.shape
        batch_size, channels, height, width = input_tensor.shape
        pool_h, pool_w = self.pooling_shape
        stride_y, stride_x = self.stride_shape

        out_h = (height - pool_h) // stride_y + 1
        out_w = (width - pool_w) // stride_x + 1

        output = np.zeros((batch_size, channels, out_h, out_w))
        self.max_positions = np.zeros((batch_size, channels, out_h, out_w, 2), dtype=int)

        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_h):
                    for j in range(out_w):
                        y0, x0 = i * stride_y, j * stride_x
                        region = input_tensor[b, c, y0:y0 + pool_h, x0:x0 + pool_w]
                        output[b, c, i, j] = np.max(region)
                        max_y, max_x = np.unravel_index(np.argmax(region), region.shape)
                        self.max_positions[b, c, i, j] = [y0 + max_y, x0 + max_x]

        return output

    def backward(self, error_tensor):
        gradient_input = np.zeros(self.input_shape)
        batch_size, channels = self.input_shape[0], self.input_shape[1]
        out_h, out_w = error_tensor.shape[2], error_tensor.shape[3]

        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_h):
                    for j in range(out_w):
                        y, x = self.max_positions[b, c, i, j]
                        gradient_input[b, c, y, x] += error_tensor[b, c, i, j]

        return gradient_input
