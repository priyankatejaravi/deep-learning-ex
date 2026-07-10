import numpy as np
import copy
from Layers.Base import BaseLayer
from Layers.FullyConnected import FullyConnected
from Layers.TanH import TanH


class RNN(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc_hidden = FullyConnected(hidden_size + input_size, hidden_size)
        self.fc_output = FullyConnected(hidden_size, output_size)
        self.tanh = TanH()

        self.trainable = True
        self.testing_phase = False
        self.hidden_state = np.zeros((1, hidden_size))
        self.memorize = False

        self._optimizer = None
        self.gradient_weights = None

    @property
    def weights(self):
        return self.fc_hidden.weights

    @weights.setter
    def weights(self, value):
        self.fc_hidden.weights = value

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        # keep a copy for each internal FC layer so they update independently
        self._optimizer = optimizer
        self.fc_hidden.optimizer = copy.deepcopy(optimizer)
        self.fc_output.optimizer = copy.deepcopy(optimizer)

    def initialize(self, weights_initializer, bias_initializer):
        self.fc_hidden.initialize(weights_initializer, bias_initializer)
        self.fc_output.initialize(weights_initializer, bias_initializer)

    def calculate_regularization_loss(self):
        if self._optimizer is not None and self._optimizer.regularizer is not None:
            return self._optimizer.regularizer.norm(self.fc_hidden.weights)
        return 0

    def forward(self, input_tensor):
        time_steps = input_tensor.shape[0]
        output_tensor = np.zeros((time_steps, self.output_size))

        if not self.memorize:
            self.hidden_state = np.zeros((1, self.hidden_size))

        # remember what happened at every time step, we need it for backward
        self.combined_list = []
        self.hidden_list = []

        for t in range(time_steps):
            x_t = input_tensor[t:t + 1, :]
            combined = np.concatenate([self.hidden_state, x_t], axis=1)
            self.combined_list.append(combined)

            pre_activation = self.fc_hidden.forward(combined)
            h_t = self.tanh.forward(pre_activation)

            self.hidden_list.append(h_t.copy())
            self.hidden_state = h_t

            y_t = self.fc_output.forward(h_t)
            output_tensor[t, :] = y_t

        return output_tensor

    def backward(self, error_tensor):
        time_steps = error_tensor.shape[0]
        gradient_input = np.zeros((time_steps, self.input_size))

        self.gradient_weights = np.zeros_like(self.fc_hidden.weights)
        gradient_weights_output = np.zeros_like(self.fc_output.weights)

        # don't let the FC layers update themselves inside the loop below,
        # we only want ONE update at the end using the gradient summed
        # over the whole sequence
        hidden_optimizer = self.fc_hidden.optimizer
        output_optimizer = self.fc_output.optimizer
        self.fc_hidden.optimizer = None
        self.fc_output.optimizer = None

        next_hidden_error = np.zeros((1, self.hidden_size))

        for t in reversed(range(time_steps)):
            self.fc_output.input_tensor = np.concatenate([self.hidden_list[t], np.ones((1, 1))], axis=1)
            error_from_output = self.fc_output.backward(error_tensor[t:t + 1, :])
            gradient_weights_output += self.fc_output.gradient_weights

            hidden_error = error_from_output + next_hidden_error

            self.tanh.activation = self.hidden_list[t]
            hidden_error = self.tanh.backward(hidden_error)

            self.fc_hidden.input_tensor = np.concatenate([self.combined_list[t], np.ones((1, 1))], axis=1)
            combined_error = self.fc_hidden.backward(hidden_error)
            self.gradient_weights += self.fc_hidden.gradient_weights

            next_hidden_error = combined_error[:, :self.hidden_size]
            gradient_input[t, :] = combined_error[:, self.hidden_size:]

        # put the optimizers back and do exactly one update each
        self.fc_hidden.optimizer = hidden_optimizer
        self.fc_output.optimizer = output_optimizer

        if self.fc_hidden.optimizer is not None:
            self.fc_hidden.weights = self.fc_hidden.optimizer.calculate_update(self.fc_hidden.weights, self.gradient_weights)
        if self.fc_output.optimizer is not None:
            self.fc_output.weights = self.fc_output.optimizer.calculate_update(self.fc_output.weights, gradient_weights_output)

        return gradient_input
