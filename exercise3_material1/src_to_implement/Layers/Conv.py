import numpy as np
import copy
from Layers.Base import BaseLayer
from scipy.ndimage import correlate, convolve
from scipy.signal import correlate as correlate1d
from scipy.signal import correlate2d


class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.trainable = True
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape  # (c, m) or (c, m, n)
        self.num_kernels = num_kernels

        self.weights = np.random.uniform(0, 1, (num_kernels, *convolution_shape))
        self.bias = np.random.uniform(0, 1, num_kernels)

        self.gradient_weights = None
        self.gradient_bias = None
        self._weights_optimizer = None
        self._bias_optimizer = None
        self.input_tensor = None

    @property
    def optimizer(self):
        return self._weights_optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._weights_optimizer = copy.deepcopy(optimizer)
        self._bias_optimizer = copy.deepcopy(optimizer)

    def initialize(self, weights_initializer, bias_initializer):
        fan_in = int(np.prod(self.convolution_shape))
        fan_out = int(np.prod(self.convolution_shape[1:]) * self.num_kernels)
        self.weights = weights_initializer.initialize((self.num_kernels, *self.convolution_shape), fan_in, fan_out)
        self.bias = bias_initializer.initialize((self.num_kernels,), fan_in, fan_out)

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        batch_size = input_tensor.shape[0]
        channels = self.convolution_shape[0]
        is_1d = len(self.convolution_shape) == 2

        if is_1d:
            stride = self.stride_shape[0]
            length = input_tensor.shape[2]
            output = np.zeros((batch_size, self.num_kernels, int(np.ceil(length / stride))))
            for b in range(batch_size):
                for k in range(self.num_kernels):
                    result = np.zeros(length)
                    for c in range(channels):
                        result += correlate(input_tensor[b, c], self.weights[k, c], mode='constant', cval=0)
                    output[b, k] = result[::stride] + self.bias[k]
        else:
            stride_y, stride_x = self.stride_shape
            height, width = input_tensor.shape[2], input_tensor.shape[3]
            output = np.zeros((batch_size, self.num_kernels, int(np.ceil(height / stride_y)), int(np.ceil(width / stride_x))))
            for b in range(batch_size):
                for k in range(self.num_kernels):
                    result = np.zeros((height, width))
                    for c in range(channels):
                        result += correlate(input_tensor[b, c], self.weights[k, c], mode='constant', cval=0)
                    output[b, k] = result[::stride_y, ::stride_x] + self.bias[k]

        return output

    def backward(self, error_tensor):
        input_tensor = self.input_tensor
        batch_size = input_tensor.shape[0]
        channels = self.convolution_shape[0]
        is_1d = len(self.convolution_shape) == 2

        if is_1d:
            stride = self.stride_shape[0]
            kernel_len = self.convolution_shape[1]
            length = input_tensor.shape[2]

            error_up = np.zeros((batch_size, self.num_kernels, length))
            error_up[:, :, ::stride] = error_tensor

            pad_before = (kernel_len - 1) // 2
            input_padded = np.pad(input_tensor, ((0, 0), (0, 0), (pad_before, kernel_len - 1 - pad_before)))

            gradient_input = np.zeros_like(input_tensor)
            for b in range(batch_size):
                for c in range(channels):
                    for k in range(self.num_kernels):
                        gradient_input[b, c] += convolve(error_up[b, k], self.weights[k, c], mode='constant', cval=0)

            self.gradient_weights = np.zeros_like(self.weights)
            for b in range(batch_size):
                for k in range(self.num_kernels):
                    for c in range(channels):
                        self.gradient_weights[k, c] += correlate1d(input_padded[b, c], error_up[b, k], mode='valid')

            self.gradient_bias = np.sum(error_tensor, axis=(0, 2))

        else:
            stride_y, stride_x = self.stride_shape
            kernel_h, kernel_w = self.convolution_shape[1], self.convolution_shape[2]
            height, width = input_tensor.shape[2], input_tensor.shape[3]

            error_up = np.zeros((batch_size, self.num_kernels, height, width))
            error_up[:, :, ::stride_y, ::stride_x] = error_tensor

            pad_h, pad_w = (kernel_h - 1) // 2, (kernel_w - 1) // 2
            input_padded = np.pad(input_tensor, ((0, 0), (0, 0), (pad_h, kernel_h - 1 - pad_h), (pad_w, kernel_w - 1 - pad_w)))

            gradient_input = np.zeros_like(input_tensor)
            for b in range(batch_size):
                for c in range(channels):
                    for k in range(self.num_kernels):
                        gradient_input[b, c] += convolve(error_up[b, k], self.weights[k, c], mode='constant', cval=0)

            self.gradient_weights = np.zeros_like(self.weights)
            for b in range(batch_size):
                for k in range(self.num_kernels):
                    for c in range(channels):
                        self.gradient_weights[k, c] += correlate2d(input_padded[b, c], error_up[b, k], mode='valid')

            self.gradient_bias = np.sum(error_tensor, axis=(0, 2, 3))

        if self._weights_optimizer is not None:
            self.weights = self._weights_optimizer.calculate_update(self.weights, self.gradient_weights)
        if self._bias_optimizer is not None:
            self.bias = self._bias_optimizer.calculate_update(self.bias, self.gradient_bias)

        return gradient_input
