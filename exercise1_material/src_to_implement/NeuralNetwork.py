import numpy as np
from Optimization import *
from Layers import *
import copy

class NeuralNetwork:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.loss = []       # stores loss value for each iteration
        self.layers = []     # holds the architecture
        self.data_layer = None   # provides input data and labels
        self.loss_layer = None   # provides loss and prediction
        self.label_tensor = None

    def forward(self):
        # get input and label from data layer
        input_tensor, self.label_tensor = self.data_layer.next()
        # pass input through all layers
        x = input_tensor
        for layer in self.layers:
            x = layer.forward(x)
        # pass through loss layer and return its output
        loss = self.loss_layer.forward(x, self.label_tensor)
        self.loss.append(loss)
        return loss

    def backward(self):
        # start from loss layer with the label tensor
        error = self.loss_layer.backward(self.label_tensor)
        # propagate back through all layers in reverse
        for layer in reversed(self.layers):
            error = layer.backward(error)

    def append_layer(self, layer):
        # if trainable, deep copy the optimizer and assign it to the layer
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
        # append to layers list (both trainable and non-trainable)
        self.layers.append(layer)

    def train(self, iterations):
        # train for given iterations and store loss each time
        for _ in range(iterations):
            self.forward()
            self.backward()

    def test(self, input_tensor):
        # propagate input through all layers, return final prediction
        x = input_tensor
        for layer in self.layers:
            x = layer.forward(x)
        return x
