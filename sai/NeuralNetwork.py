import numpy as np
from Optimization import *
from Layers import *
import copy

class NeuralNetwork:
    """Neural Network for training and inference."""
    
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.current_labels = None

    def forward(self):
        """Forward pass: compute predictions and loss."""
        # Get data batch
        input_data, self.current_labels = self.data_layer.next()
        
        # Forward through network
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        
        # Compute loss
        loss = self.loss_layer.forward(output, self.current_labels)
        self.loss.append(loss)
        return loss

    def backward(self):
        """Backward pass: compute gradients and update weights."""
        # Start backprop from loss
        error = self.loss_layer.backward(self.current_labels)
        
        # Back through layers
        for layer in reversed(self.layers):
            error = layer.backward(error)

    def append_layer(self, layer):
        """Add layer to network."""
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
        self.layers.append(layer)

    def train(self, iterations):
        """Train network."""
        for _ in range(iterations):
            self.forward()
            self.backward()

    def test(self, input_data):
        """Inference mode."""
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output
