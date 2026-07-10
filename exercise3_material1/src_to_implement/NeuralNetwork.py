import copy
import pickle


class NeuralNetwork:
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
        if self.layers:
            return self.layers[0].testing_phase
        return False

    @phase.setter
    def phase(self, testing_phase):
        for layer in self.layers:
            layer.testing_phase = testing_phase

    def forward(self):
        input_tensor, self.current_labels = self.data_layer.next()

        output_tensor = input_tensor
        for layer in self.layers:
            output_tensor = layer.forward(output_tensor)

        data_loss = self.loss_layer.forward(output_tensor, self.current_labels)

        regularization_loss = 0
        for layer in self.layers:
            if layer.trainable and layer.optimizer is not None and layer.optimizer.regularizer is not None:
                regularization_loss += layer.optimizer.regularizer.norm(layer.weights)

        total_loss = data_loss + regularization_loss
        self.loss.append(total_loss)
        return total_loss

    def backward(self):
        error_tensor = self.loss_layer.backward(self.current_labels)
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)

    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
            layer.initialize(self.weights_initializer, self.bias_initializer)
        self.layers.append(layer)

    def train(self, iterations):
        self.phase = False
        for _ in range(iterations):
            self.forward()
            self.backward()

    def test(self, input_tensor):
        self.phase = True
        output_tensor = input_tensor
        for layer in self.layers:
            output_tensor = layer.forward(output_tensor)
        return output_tensor

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
