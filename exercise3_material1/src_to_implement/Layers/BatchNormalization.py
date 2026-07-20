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
        # This layer has trainable gamma and beta parameters
        self.trainable = True
        self.channels = channels

        # Separate optimizers for weights (gamma) and bias (beta)
        self._weights_optimizer = None
        self._bias_optimizer = None

        # Running stats for test-time inference (initialized on first batch)
        self.mean = None
        self.var = None
        self.running_mean = None
        self.running_var = None

        # Saved values from forward pass needed for backward
        self.input_tensor = None
        self.x_hat = None
        self.image_shape = None

        # Initialize weights and bias properly
        self.initialize(None, None)

    # Function entry point
    def initialize(self, weights_initializer, bias_initializer):
        # gamma starts at 1 so normalization has no effect initially
        self.weights = np.ones(self.channels)
        # beta starts at 0 for the same reason
        self.bias = np.zeros(self.channels)

    @property
    # Function entry point
    def optimizer(self):
        return self._weights_optimizer

    @optimizer.setter
    # Function entry point
    def optimizer(self, optimizer):
        # We need two separate optimizer instances because gamma and beta
        # accumulate separate moment estimates (for Adam etc.)
        self._weights_optimizer = optimizer
        self._bias_optimizer = copy.deepcopy(optimizer)

    # Function entry point
    def reformat(self, tensor):
        """Reshape between image (4D) and vector (2D) format.

        4D (B, C, H, W)  →  2D (B*H*W, C)   — flatten spatial dims
        2D (B*H*W, C)    →  4D (B, C, H, W)  — restore original shape
        """
        if tensor.ndim == 4:
            # Store the original shape so we can reverse this later
            self.image_shape = tensor.shape
            batch, channels, height, width = tensor.shape
            # Move channels to last, then flatten batch+spatial together
            tensor = tensor.transpose(0, 2, 3, 1).reshape(batch * height * width, channels)
        else:
            # We must have the original 4D shape stored from the forward pass
            batch, channels, height, width = self.image_shape
            # Reverse: unflatten then move channels back to dim 1            
            tensor = tensor.reshape(batch, height, width, channels).transpose(0, 3, 1, 2)
        return tensor

    # Function entry point
    def forward(self, input_tensor):
        # If the input is 4D (image), reformat to 2D for uniform processing
        is_image = input_tensor.ndim == 4
        if is_image:
            input_tensor = self.reformat(input_tensor)

        # Save the (possibly reformatted) input for the backward pass
        self.input_tensor = input_tensor
        eps = 1e-15 # small value to avoid dividing by zero

        if not self.testing_phase:
            # ---- Training: normalize using current batch statistics ----
            self.mean = np.mean(input_tensor, axis=0)
            self.var = np.var(input_tensor, axis=0)

            # Initialize running stats on the very first batch
            if self.running_mean is None:
                self.running_mean = self.mean.copy()
                self.running_var = self.var.copy()
            else:
                # Exponential moving average to track population statistics
                self.running_mean = 0.8 * self.running_mean + 0.2 * self.mean
                self.running_var = 0.8 * self.running_var + 0.2 * self.var
            
            # Normalize, scale and shift
            self.x_hat = (input_tensor - self.mean) / np.sqrt(self.var + eps)
        else:
            # ---- Testing: use the running stats from training ----
            self.x_hat = (input_tensor - self.running_mean) / np.sqrt(self.running_var + eps)

        output = self.weights * self.x_hat + self.bias

        # Restore 4D shape for conv case
        if is_image:
            output = self.reformat(output)

        return output


    # Function entry point
    def backward(self, error_tensor):
        # If the error comes in as 4D, reformat to 2D
        is_image = error_tensor.ndim == 4
        if is_image:
            error_tensor = self.reformat(error_tensor)

        # Gradient w.r.t. gamma (sum over batch dimension)
        self.gradient_weights = np.sum(error_tensor * self.x_hat, axis=0)
        # Gradient w.r.t. beta (sum over batch dimension)
        self.gradient_bias = np.sum(error_tensor, axis=0)

        # Gradient w.r.t. input (use the provided helper)
        gradient_input = compute_bn_gradients(error_tensor, self.input_tensor, self.weights, self.mean, self.var, 1e-15)

        # Update gamma and beta if optimizers are set
        if self._weights_optimizer is not None:
            self.weights = self._weights_optimizer.calculate_update(self.weights, self.gradient_weights)
        if self._bias_optimizer is not None:
            self.bias = self._bias_optimizer.calculate_update(self.bias, self.gradient_bias)

        # Restore 4D shape for conv case
        if is_image:
            gradient_input = self.reformat(gradient_input)

        return gradient_input
