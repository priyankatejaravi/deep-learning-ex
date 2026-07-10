import numpy as np
import copy
from Layers.Base import BaseLayer
from Layers.FullyConnected import FullyConnected
from Layers.TanH import TanH
from Layers.Sigmoid import Sigmoid

class LSTM(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self._memorize = False
        self._optimizer = None
        
        self.fc_f = FullyConnected(hidden_size + input_size, hidden_size)
        self.fc_i = FullyConnected(hidden_size + input_size, hidden_size)
        self.fc_c = FullyConnected(hidden_size + input_size, hidden_size)
        self.fc_o = FullyConnected(hidden_size + input_size, hidden_size)
        self.fc_output = FullyConnected(hidden_size, output_size)
        
        self.hidden_state = np.zeros((1, hidden_size))
        self.cell_state = np.zeros((1, hidden_size))
        self._gradient_weights = None

    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, value):
        self._memorize = value

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, opt):
        self._optimizer = opt
        self.fc_f.optimizer = copy.deepcopy(opt)
        self.fc_i.optimizer = copy.deepcopy(opt)
        self.fc_c.optimizer = copy.deepcopy(opt)
        self.fc_o.optimizer = copy.deepcopy(opt)
        self.fc_output.optimizer = copy.deepcopy(opt)

    @property
    def weights(self):
        # Return something to satisfy property if needed.
        return self.fc_f.weights

    @weights.setter
    def weights(self, value):
        self.fc_f.weights = value

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value

    def initialize(self, weights_initializer, bias_initializer):
        self.fc_f.initialize(weights_initializer, bias_initializer)
        self.fc_i.initialize(weights_initializer, bias_initializer)
        self.fc_c.initialize(weights_initializer, bias_initializer)
        self.fc_o.initialize(weights_initializer, bias_initializer)
        self.fc_output.initialize(weights_initializer, bias_initializer)

    def calculate_regularization_loss(self):
        loss = 0
        if self._optimizer is not None and self._optimizer.regularizer is not None:
            loss += self._optimizer.regularizer.norm(self.fc_f.weights)
            loss += self._optimizer.regularizer.norm(self.fc_i.weights)
            loss += self._optimizer.regularizer.norm(self.fc_c.weights)
            loss += self._optimizer.regularizer.norm(self.fc_o.weights)
        return loss

    def forward(self, input_tensor):
        time_steps = input_tensor.shape[0]
        output_tensor = np.zeros((time_steps, self.output_size))

        if not self._memorize:
            self.hidden_state = np.zeros((1, self.hidden_size))
            self.cell_state = np.zeros((1, self.hidden_size))

        self.saved_states = []

        sigmoid = Sigmoid()
        tanh = TanH()

        for t in range(time_steps):
            x_t = input_tensor[t:t+1, :]
            combined = np.concatenate([self.hidden_state, x_t], axis=1)

            f_t = sigmoid.forward(self.fc_f.forward(combined))
            i_t = sigmoid.forward(self.fc_i.forward(combined))
            c_tilde_t = tanh.forward(self.fc_c.forward(combined))
            o_t = sigmoid.forward(self.fc_o.forward(combined))

            prev_c = self.cell_state.copy()
            self.cell_state = f_t * self.cell_state + i_t * c_tilde_t
            
            tanh_c_t = tanh.forward(self.cell_state)
            self.hidden_state = o_t * tanh_c_t
            
            y_t = self.fc_output.forward(self.hidden_state)
            output_tensor[t, :] = y_t

            self.saved_states.append({
                'combined': combined,
                'f_t': f_t,
                'i_t': i_t,
                'c_tilde_t': c_tilde_t,
                'o_t': o_t,
                'prev_c': prev_c,
                'c_t': self.cell_state.copy(),
                'tanh_c_t': tanh_c_t,
                'h_t': self.hidden_state.copy()
            })

        return output_tensor

    def backward(self, error_tensor):
        time_steps = error_tensor.shape[0]
        dx_total = np.zeros((time_steps, self.input_size))

        grad_Wf = np.zeros_like(self.fc_f.weights)
        grad_Wi = np.zeros_like(self.fc_i.weights)
        grad_Wc = np.zeros_like(self.fc_c.weights)
        grad_Wo = np.zeros_like(self.fc_o.weights)

        dh_next = np.zeros((1, self.hidden_size))
        dc_next = np.zeros((1, self.hidden_size))

        for t in reversed(range(time_steps)):
            state = self.saved_states[t]
            combined = state['combined']
            f_t, i_t, c_tilde_t, o_t = state['f_t'], state['i_t'], state['c_tilde_t'], state['o_t']
            prev_c, c_t, tanh_c_t, h_t = state['prev_c'], state['c_t'], state['tanh_c_t'], state['h_t']

            self.fc_output.input_tensor = np.concatenate([h_t, np.ones((1, 1))], axis=1)
            e_out = error_tensor[t:t+1, :]
            dh_from_output = self.fc_output.backward(e_out)

            dh = dh_from_output + dh_next

            do = dh * tanh_c_t
            d_tanh_c = dh * o_t
            dc = d_tanh_c * (1 - tanh_c_t**2) + dc_next

            df = dc * prev_c
            di = dc * c_tilde_t
            dc_tilde = dc * i_t

            # backward through activations
            do_pre = do * o_t * (1 - o_t)
            df_pre = df * f_t * (1 - f_t)
            di_pre = di * i_t * (1 - i_t)
            dc_tilde_pre = dc_tilde * (1 - c_tilde_t**2)

            self.fc_o.input_tensor = np.concatenate([combined, np.ones((1, 1))], axis=1)
            self.fc_f.input_tensor = np.concatenate([combined, np.ones((1, 1))], axis=1)
            self.fc_i.input_tensor = np.concatenate([combined, np.ones((1, 1))], axis=1)
            self.fc_c.input_tensor = np.concatenate([combined, np.ones((1, 1))], axis=1)

            d_combined_o = self.fc_o.backward(do_pre)
            d_combined_f = self.fc_f.backward(df_pre)
            d_combined_i = self.fc_i.backward(di_pre)
            d_combined_c = self.fc_c.backward(dc_tilde_pre)

            grad_Wo += self.fc_o.gradient_weights
            grad_Wf += self.fc_f.gradient_weights
            grad_Wi += self.fc_i.gradient_weights
            grad_Wc += self.fc_c.gradient_weights

            d_combined = d_combined_o + d_combined_f + d_combined_i + d_combined_c
            dh_next = d_combined[:, :self.hidden_size]
            dx_total[t, :] = d_combined[:, self.hidden_size:]
            dc_next = dc * f_t

        if self.fc_f.optimizer is not None:
            self.fc_f.weights = self.fc_f.optimizer.calculate_update(self.fc_f.weights, grad_Wf)
            self.fc_i.weights = self.fc_i.optimizer.calculate_update(self.fc_i.weights, grad_Wi)
            self.fc_c.weights = self.fc_c.optimizer.calculate_update(self.fc_c.weights, grad_Wc)
            self.fc_o.weights = self.fc_o.optimizer.calculate_update(self.fc_o.weights, grad_Wo)

        self._gradient_weights = grad_Wf

        return dx_total
