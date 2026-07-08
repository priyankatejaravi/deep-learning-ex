import numpy as np
from Optimization import *
from Layers import *
import copy
import pickle

class NeuralNetwork:
    """Neural Network for training and inference."""

    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.optimizer = optimizer

        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer

        self.loss = []    
        self.layers = []  
        self.data_layer = None 
        self.loss_layer = None 
        self.current_labels = None

    @property
    def phase(self):
        return self.layers[0].testing_phase if self.layers else False

    @phase.setter
    def phase(self, testing):
        for layer in self.layers:
            layer.testing_phase = testing

    def forward(self):
        """Forward pass: compute predictions and loss."""

        # Get data batch
        input_tensor, self.current_labels = self.data_layer.next()

        # Forward through network
        output = input_tensor
        for layer in self.layers:
            output = layer.forward(output)

        # Compute data loss
        data_loss = self.loss_layer.forward(output, self.current_labels)
        
        # Add regularization loss
        reg_loss = 0.0
        for layer in self.layers:
            if layer.trainable and layer.optimizer is not None and layer.optimizer.regularizer is not None:
                reg_loss += layer.optimizer.regularizer.norm(layer.weights)
                
        total_loss = data_loss + reg_loss
        self.loss.append(total_loss)
        return total_loss

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
            # initialize weights with the stored initializers
            layer.initialize(self.weights_initializer, self.bias_initializer)
        self.layers.append(layer)

    def train(self, iterations):
        """Train network."""
        self.phase = False
        for _ in range(iterations):
            self.forward()
            self.backward()

    def test(self, input_tensor):
        """Inference mode."""
        self.phase = True
        x = input_tensor
        for layer in self.layers:
            x = layer.forward(x)
        self.phase = False
        return x

    def __getstate__(self):
        state = self.__dict__.copy()
        state['data_layer'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


def save(filename, net):
    with open(filename, 'wb') as f:
        pickle.dump(net, f)


def load(filename, data_layer):
    with open(filename, 'rb') as f:
        net = pickle.load(f)
    net.data_layer = data_layer
    return net
