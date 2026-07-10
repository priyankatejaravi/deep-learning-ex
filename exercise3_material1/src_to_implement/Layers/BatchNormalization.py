import numpy as np
import copy
from Layers.Base import BaseLayer
from Layers.Helpers import compute_bn_gradients


class BatchNormalization(BaseLayer):
    """Batch Normalization layer for both FC (vector) and Conv (image) inputs.

    For vector inputs shape is (batch, channels).
    For image inputs shape is (batch, channels, height, width).

    The layer normalizes over the batch (and spatial dims for images),
    then applies learned scale gamma and shift beta.
    """

    # Function entry point
    def __init__(self, channels):
        super().__init__()
        self.trainable = True
        self.channels = channels

        self._weights_optimizer = None
        self._bias_optimizer = None

        self.mean = None
        self.var = None
        self.running_mean = None
        self.running_var = None

        self.input_tensor = None
        self.x_hat = None
        self.image_shape = None

        self.initialize(None, None)

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = np.ones(self.channels)
        self.bias = np.zeros(self.channels)

    @property
    def optimizer(self):
        return self._weights_optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._weights_optimizer = optimizer
        self._bias_optimizer = copy.deepcopy(optimizer)

    def reformat(self, tensor):
        if tensor.ndim == 4:
            self.image_shape = tensor.shape
            batch, channels, height, width = tensor.shape
            tensor = tensor.transpose(0, 2, 3, 1).reshape(batch * height * width, channels)
        else:
            batch, channels, height, width = self.image_shape
            tensor = tensor.reshape(batch, height, width, channels).transpose(0, 3, 1, 2)
        return tensor

    def forward(self, input_tensor):
        is_image = input_tensor.ndim == 4
        if is_image:
            input_tensor = self.reformat(input_tensor)

        self.input_tensor = input_tensor
        eps = 1e-15

        if not self.testing_phase:
            self.mean = np.mean(input_tensor, axis=0)
            self.var = np.var(input_tensor, axis=0)

            if self.running_mean is None:
                self.running_mean = self.mean.copy()
                self.running_var = self.var.copy()
            else:
                self.running_mean = 0.8 * self.running_mean + 0.2 * self.mean
                self.running_var = 0.8 * self.running_var + 0.2 * self.var

            self.x_hat = (input_tensor - self.mean) / np.sqrt(self.var + eps)
        else:
            self.x_hat = (input_tensor - self.running_mean) / np.sqrt(self.running_var + eps)

        output = self.weights * self.x_hat + self.bias

        if is_image:
            output = self.reformat(output)

        return output

    def backward(self, error_tensor):
        is_image = error_tensor.ndim == 4
        if is_image:
            error_tensor = self.reformat(error_tensor)

        self.gradient_weights = np.sum(error_tensor * self.x_hat, axis=0)
        self.gradient_bias = np.sum(error_tensor, axis=0)

        gradient_input = compute_bn_gradients(error_tensor, self.input_tensor, self.weights, self.mean, self.var, 1e-15)

        if self._weights_optimizer is not None:
            self.weights = self._weights_optimizer.calculate_update(self.weights, self.gradient_weights)
        if self._bias_optimizer is not None:
            self.bias = self._bias_optimizer.calculate_update(self.bias, self.gradient_bias)

        if is_image:
            gradient_input = self.reformat(gradient_input)

        return gradient_input
