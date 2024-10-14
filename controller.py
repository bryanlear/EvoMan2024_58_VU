import numpy as np
from evoman.controller import Controller

# Activation functions
def ReLu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

class player_controller(Controller):
    def __init__(self, n_hidden):
        self.n_hidden = n_hidden
        self.hidden_state = np.zeros(self.n_hidden)  # Hidden state

    def get_n_params(self, n_inputs):
        n_hidden = self.n_hidden

        # Weights and biases for input to hidden layer
        n_params = n_inputs * n_hidden      # W_xh
        n_params += n_hidden * n_hidden     # W_hh
        n_params += n_hidden                # b_h

        # Weights and biases for hidden to output layer
        n_params += n_hidden * 5            # W_hy (5 outputs)
        n_params += 5                       # b_y

        return n_params

    def set_weights(self, weights, n_inputs):
        n_hidden = self.n_hidden
        idx = 0

        # Input to hidden weights
        self.W_xh = weights[idx:idx + n_inputs * n_hidden].reshape(n_inputs, n_hidden)
        idx += n_inputs * n_hidden

        # Hidden to hidden weights
        self.W_hh = weights[idx:idx + n_hidden * n_hidden].reshape(n_hidden, n_hidden)
        idx += n_hidden * n_hidden

        # Hidden biases
        self.b_h = weights[idx:idx + n_hidden]
        idx += n_hidden

        # Hidden to output weights
        self.W_hy = weights[idx:idx + n_hidden * 5].reshape(n_hidden, 5)
        idx += n_hidden * 5

        # Output biases
        self.b_y = weights[idx:idx + 5]
        idx += 5

    def reset(self):
        # Reset hidden state at the beginning of each episode
        self.hidden_state = np.zeros(self.n_hidden)

    def control(self, inputs, controller):
        # Normalize inputs
        inputs = (inputs - np.min(inputs)) / (np.max(inputs) - np.min(inputs) + 1e-8)

        # Recurrent neural network computation
        h_input = np.dot(inputs, self.W_xh) + np.dot(self.hidden_state, self.W_hh) + self.b_h
        self.hidden_state = ReLu(h_input)

        o_input = np.dot(self.hidden_state, self.W_hy) + self.b_y
        output = tanh(o_input)

        # Decision thresholds
        actions = [1 if o > 0 else 0 for o in output]
        return actions[:5]