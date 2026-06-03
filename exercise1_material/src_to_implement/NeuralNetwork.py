import numpy as np
from Optimization import *
from Layers import *
import copy

class NeuralNetwork:
    """Neural Network for training and inference."""

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.loss = []       # stores loss value for each iteration
        self.layers = []     # holds the architecture
        self.data_layer = None   # provides input data and labels
        self.loss_layer = None   # provides loss and prediction
        self.current_labels = None

    def forward(self):
        """Forward pass: compute predictions and loss."""

        # get input and label from data layer
        input_tensor, self.current_labels = self.data_layer.next()
        # pass input through all layers
        output = input_tensor
        for layer in self.layers:
            output = layer.forward(output)

        # pass through loss layer and return its output
        loss = self.loss_layer.forward(output, self.current_labels)
        self.loss.append(loss)
        return loss

    def backward(self):
        """Backward pass: compute gradients and update weights."""

        # start from loss layer with the label tensor
        error = self.loss_layer.backward(self.current_labels)
        # propagate back through all layers in reverse
        for layer in reversed(self.layers):
            error = layer.backward(error)

    def append_layer(self, layer):
        """Add layer to network."""

        # if trainable, deep copy the optimizer and assign it to the layer
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
        # append to layers list (both trainable and non-trainable)
        self.layers.append(layer)

    def train(self, iterations):
        """Train network."""
        for _ in range(iterations):
            self.forward()
            self.backward()

    def test(self, input_tensor):
        """Inference mode."""
        x = input_tensor
        for layer in self.layers:
            x = layer.forward(x)
        return x
