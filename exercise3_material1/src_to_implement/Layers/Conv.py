# CNN convolution layer with forward/backprop and optimizer handling
import numpy as np
import copy
from Layers.Base import BaseLayer

from scipy.ndimage import correlate, convolve
from scipy.signal import correlate as correlate1d
from scipy.signal import correlate2d


"""Convolutional layer implementation.

Supports 1D and 2D same-correlation (zero padding) with configurable stride.
Implements forward, backward and parameter updates using per-parameter optimizers.
"""


# Main component implementation
class Conv(BaseLayer):

    # constructor receiving stride_shape, convolution_shape, num_kernels
    # Function entry point
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.trainable = True
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape  # (c, m) for 1D or (c, m, n) for 2D
        self.num_kernels = num_kernels
        # weights and bias initialized uniformly random in [0, 1)
        self.weights = np.random.uniform(0, 1, (num_kernels, *convolution_shape))
        self.bias = np.random.uniform(0, 1, num_kernels)
        self._gradient_weights = None
        self._gradient_bias = None
        self._weights_optimizer = None
        self._bias_optimizer = None
        self._input_tensor = None
    
    
    @property
    # Function entry point
    def gradient_weights(self):
        return self._gradient_weights

    @property
    # Function entry point
    def gradient_bias(self):
        return self._gradient_bias
    
    # two copies of optimizer needed: one for weights, one for bias
    @property
    # Function entry point
    def optimizer(self):
        return self._weights_optimizer

    @optimizer.setter
    # Function entry point
    def optimizer(self, optimizer):
        self._weights_optimizer = copy.deepcopy(optimizer)
        self._bias_optimizer = copy.deepcopy(optimizer)

    # Function entry point
    def initialize(self, weights_initializer, bias_initializer):
        # fan_in = input_channels * kernel_height * kernel_width
        fan_in = int(np.prod(self.convolution_shape))
        # fan_out = output_channels * kernel_height * kernel_width
        fan_out = int(np.prod(self.convolution_shape[1:]) * self.num_kernels)
        self.weights = weights_initializer.initialize(
            (self.num_kernels, *self.convolution_shape), fan_in, fan_out
        )
        self.bias = bias_initializer.initialize((self.num_kernels,), fan_in, fan_out)

    # Function entry point
    def forward(self, input_tensor):
        self._input_tensor = input_tensor
        B = input_tensor.shape[0]
        C = self.convolution_shape[0]
        is_1d = len(self.convolution_shape) == 2

        if is_1d:
            sy = self.stride_shape[0]
            L = input_tensor.shape[2]
            output = np.zeros((B, self.num_kernels, int(np.ceil(L / sy))))
            for b in range(B):
                for k in range(self.num_kernels):
                    result = np.zeros(L)
                    for c in range(C):
                        # cross-correlation with same padding (zero padding)
                        result += correlate(input_tensor[b, c], self.weights[k, c], mode='constant', cval=0)
                    # subsample with stride, add bias
                    output[b, k] = result[::sy] + self.bias[k]
        else:
            sy, sx = self.stride_shape
            H, W = input_tensor.shape[2], input_tensor.shape[3]
            output = np.zeros((B, self.num_kernels, int(np.ceil(H / sy)), int(np.ceil(W / sx))))
            for b in range(B):
                for k in range(self.num_kernels):
                    result = np.zeros((H, W))
                    for c in range(C):
                        # cross-correlation with same padding (zero padding)
                        result += correlate(input_tensor[b, c], self.weights[k, c], mode='constant', cval=0)
                    # subsample with stride, add bias
                    output[b, k] = result[::sy, ::sx] + self.bias[k]

        return output

    # Function entry point
    def backward(self, error_tensor):
        input_tensor = self._input_tensor
        B = input_tensor.shape[0]
        C = self.convolution_shape[0]
        is_1d = len(self.convolution_shape) == 2

        if is_1d:
            sy = self.stride_shape[0]
            kL = self.convolution_shape[1]
            L = input_tensor.shape[2]

            # upsample error by inserting zeros between elements for stride > 1
            error_up = np.zeros((B, self.num_kernels, L))
            error_up[:, :, ::sy] = error_tensor

            # same padding as used in forward
            pad_b = (kL - 1) // 2
            input_padded = np.pad(input_tensor, ((0,0),(0,0),(pad_b, kL-1-pad_b)))

            # gradient w.r.t. input: use convolution (flips kernel = backward of correlation)
            error_input = np.zeros_like(input_tensor)
            for b in range(B):
                for c in range(C):
                    for k in range(self.num_kernels):
                        error_input[b, c] += convolve(error_up[b, k], self.weights[k, c], mode='constant', cval=0)

            # gradient w.r.t. weights: valid correlation of padded input with upsampled error
            self._gradient_weights = np.zeros_like(self.weights)
            for b in range(B):
                for k in range(self.num_kernels):
                    for c in range(C):
                        self._gradient_weights[k, c] += correlate1d(input_padded[b, c], error_up[b, k], mode='valid')

            # gradient w.r.t. bias: sum over batch and spatial dimensions
            self._gradient_bias = np.sum(error_tensor, axis=(0, 2))

        else:
            sy, sx = self.stride_shape
            kH, kW = self.convolution_shape[1], self.convolution_shape[2]
            H, W = input_tensor.shape[2], input_tensor.shape[3]

            # upsample error by inserting zeros between elements for stride > 1
            error_up = np.zeros((B, self.num_kernels, H, W))
            error_up[:, :, ::sy, ::sx] = error_tensor

            # same padding as used in forward
            pad_h, pad_w = (kH - 1) // 2, (kW - 1) // 2
            input_padded = np.pad(input_tensor, ((0,0),(0,0),(pad_h, kH-1-pad_h),(pad_w, kW-1-pad_w)))

            # gradient w.r.t. input: use convolution (flips kernel = backward of correlation)
            error_input = np.zeros_like(input_tensor)
            for b in range(B):
                for c in range(C):
                    for k in range(self.num_kernels):
                        error_input[b, c] += convolve(error_up[b, k], self.weights[k, c], mode='constant', cval=0)

            # gradient w.r.t. weights: valid correlation of padded input with upsampled error
            self._gradient_weights = np.zeros_like(self.weights)
            for b in range(B):
                for k in range(self.num_kernels):
                    for c in range(C):
                        self._gradient_weights[k, c] += correlate2d(input_padded[b, c], error_up[b, k], mode='valid')

            # gradient w.r.t. bias: sum over batch, height, width
            self._gradient_bias = np.sum(error_tensor, axis=(0, 2, 3))

        # update weights and bias with their own optimizer copies
        if self._weights_optimizer is not None:
            self.weights = self._weights_optimizer.calculate_update(self.weights, self._gradient_weights)
        if self._bias_optimizer is not None:
            self.bias = self._bias_optimizer.calculate_update(self.bias, self._gradient_bias)

        return error_input


