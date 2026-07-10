# Elman Recurrent Neural Network layer
import numpy as np
import copy
from Layers.Base import BaseLayer
from Layers.FullyConnected import FullyConnected
from Layers.TanH import TanH


# Main component implementation
class RNN(BaseLayer):
    """Elman RNN layer with Backpropagation Through Time (BPTT).

    Architecture per time step:
        combined  = [h_{t-1}, x_t]            (hidden + input stacked)
        h_t       = tanh( FC_hidden(combined) )
        y_t       = FC_output(h_t)

    The "batch" dimension of the input tensor is treated as the time dimension.
    """

    # Function entry point
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.trainable = True

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Internal FC layer that computes the new hidden state from [h, x]
        self.fc_hidden = FullyConnected(hidden_size + input_size, hidden_size)

        # Internal FC layer that maps the hidden state to the output
        self.fc_output = FullyConnected(hidden_size, output_size)

        # TanH activation applied after the hidden FC layer
        self.tanh = TanH()

        # Hidden state — starts at zero
        self.hidden_state = np.zeros((1, hidden_size))

        # When memorize=True, the hidden state is carried over between sequences
        self._memorize = False

        # Will hold the optimizer assigned from outside
        self._optimizer = None

        # Accumulated gradient w.r.t. the hidden FC weights (across time steps)
        self._gradient_weights = None

    @property
    # Function entry point
    def memorize(self):
        return self._memorize

    @memorize.setter
    # Function entry point
    def memorize(self, value):
        self._memorize = value

    @property
    # Function entry point
    def weights(self):
        # "The weights" of an RNN are the weights of the hidden FC layer
        return self.fc_hidden.weights

    @weights.setter
    # Function entry point
    def weights(self, value):
        self.fc_hidden.weights = value

    @property
    # Function entry point
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    # Function entry point
    def gradient_weights(self, value):
        self._gradient_weights = value

    @property
    # Function entry point
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    # Function entry point
    def optimizer(self, opt):
        # Copy to fc_hidden and fc_output so both get updated
        self._optimizer = opt
        self.fc_hidden.optimizer = copy.deepcopy(opt)
        self.fc_output.optimizer = copy.deepcopy(opt)

    # Function entry point
    def initialize(self, weights_initializer, bias_initializer):
        # Initialize both internal FC layers with the given initializers
        self.fc_hidden.initialize(weights_initializer, bias_initializer)
        self.fc_output.initialize(weights_initializer, bias_initializer)

    # Function entry point
    def calculate_regularization_loss(self):
        """Return regularization norm for the hidden FC weights."""
        if self._optimizer is not None and self._optimizer.regularizer is not None:
            return self._optimizer.regularizer.norm(self.fc_hidden.weights)
        return 0

    # Function entry point
    def forward(self, input_tensor):
        """Run the RNN over a sequence.

        input_tensor shape: (time_steps, input_size)
        Returns output_tensor shape: (time_steps, output_size)
        """
        time_steps = input_tensor.shape[0]
        output_tensor = np.zeros((time_steps, self.output_size))

        # Decide starting hidden state
        if not self._memorize:
            # Fresh start — reset hidden state to zeros each sequence
            self.hidden_state = np.zeros((1, self.hidden_size))

        # Save per-step inputs for backprop (need [h, x] at each step)
        self.saved_combined = []      # stacked [h_{t-1}, x_t] before FC_hidden
        self.saved_hidden = []        # h_t after tanh at each step
        self.saved_h_before_tanh = [] # hidden FC output before tanh (used by tanh backward)

        for t in range(time_steps):
            x_t = input_tensor[t:t+1, :]  # shape (1, input_size)

            # Stack previous hidden state with current input
            combined = np.concatenate([self.hidden_state, x_t], axis=1)  # (1, hidden+input)
            self.saved_combined.append(combined)

            # Hidden FC: combined → pre_tanh
            pre_tanh = self.fc_hidden.forward(combined)  # (1, hidden_size)

            # TanH activation
            h_t = self.tanh.forward(pre_tanh)  # (1, hidden_size)

            self.saved_hidden.append(h_t.copy())
            self.hidden_state = h_t

            # Output FC: h_t → y_t
            y_t = self.fc_output.forward(h_t)  # (1, output_size)
            output_tensor[t, :] = y_t

        return output_tensor

    # Function entry point
    def backward(self, error_tensor):
        """BPTT: unroll gradients through time steps.

        error_tensor shape: (time_steps, output_size)
        Returns gradient w.r.t. input, shape: (time_steps, input_size)
        """
        time_steps = error_tensor.shape[0]
        dx_total = np.zeros((time_steps, self.input_size))

        # Accumulated weight gradient for the hidden FC
        self._gradient_weights = np.zeros_like(self.fc_hidden.weights)

        # Gradient flowing back from the next time step's hidden state
        dh_next = np.zeros((1, self.hidden_size))

        for t in reversed(range(time_steps)):
            # --- Output layer backward ---
            # Restore the hidden state that was used at this step as input to fc_output
            self.fc_output.input_tensor = np.concatenate(
                [self.saved_hidden[t], np.ones((1, 1))], axis=1
            )
            # Error at output for this time step
            e_out = error_tensor[t:t+1, :]
            # fc_output backward gives gradient w.r.t. h_t from output
            dh_from_output = self.fc_output.backward(e_out)  # (1, hidden_size)

            # Combine gradient from output and gradient from next hidden state
            dh = dh_from_output + dh_next

            # --- TanH backward ---
            # Restore tanh activations for this step
            self.tanh.activations = self.saved_hidden[t]
            dh_pre_tanh = self.tanh.backward(dh)  # (1, hidden_size)

            # --- Hidden FC backward ---
            # Restore the combined input [h_{t-1}, x_t] for this step
            self.fc_hidden.input_tensor = np.concatenate(
                [self.saved_combined[t], np.ones((1, 1))], axis=1
            )
            d_combined = self.fc_hidden.backward(dh_pre_tanh)  # (1, hidden+input)

            # Accumulate weight gradients (don't let fc_hidden update weights yet)
            self._gradient_weights += self.fc_hidden.gradient_weights

            # Split d_combined back into dh_prev and dx
            dh_next = d_combined[:, :self.hidden_size]
            dx_t = d_combined[:, self.hidden_size:]

            dx_total[t, :] = dx_t

        # Now do one weight update for fc_hidden using the accumulated gradient
        if self.fc_hidden.optimizer is not None:
            self.fc_hidden.weights = self.fc_hidden.optimizer.calculate_update(
                self.fc_hidden.weights, self._gradient_weights
            )

        return dx_total
