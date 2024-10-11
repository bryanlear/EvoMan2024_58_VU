
import numpy as np
from evoman.controller import Controller

# Activation functions
def ReLu(x):
    return np.maximum(0, x)

def Sigmoid(x):
    return 1 / (1 + np.exp(-x))

class player_controller(Controller):
    def __init__(self, n_hidden):
        self.n_hidden = n_hidden

    def get_n_params(self, n_inputs):
        # Calculate the total number of parameters (weights and biases)
        n_hidden = self.n_hidden
        n_params = 0

        if n_hidden > 0:
            # Weights and biases between input and hidden layer
            n_params += n_inputs * n_hidden + n_hidden
            # Weights and biases between hidden and output layer
            n_params += n_hidden * 5 + 5  # 5 outputs
        else:
            # Weights and biases between input and output layer
            n_params += n_inputs * 5 + 5

        return n_params

    def set_weights(self, weights, n_inputs):
        # Set the weights and biases from the flattened weights array
        n_hidden = self.n_hidden

        if n_hidden > 0:
            # Weights and biases for input to hidden layer
            idx = 0
            self.weights1 = weights[idx:idx + n_inputs * n_hidden].reshape(n_inputs, n_hidden)
            idx += n_inputs * n_hidden
            self.bias1 = weights[idx:idx + n_hidden].reshape(1, n_hidden)
            idx += n_hidden

            # Weights and biases for hidden to output layer
            self.weights2 = weights[idx:idx + n_hidden * 5].reshape(n_hidden, 5)
            idx += n_hidden * 5
            self.bias2 = weights[idx:idx + 5].reshape(1, 5)
            idx += 5
        else:
            # Weights and biases for input to output layer
            idx = 0
            self.weights = weights[idx:idx + n_inputs * 5].reshape(n_inputs, 5)
            idx += n_inputs * 5
            self.bias = weights[idx:idx + 5].reshape(1, 5)
            idx += 5

    def control(self, inputs, controller):
        # Normalize inputs
        inputs = (inputs - np.min(inputs)) / (np.max(inputs) - np.min(inputs) + 1e-8)

        if self.n_hidden > 0:
            # Forward pass through hidden layer
            hidden = ReLu(np.dot(inputs, self.weights1) + self.bias1)
            # Forward pass through output layer
            output = Sigmoid(np.dot(hidden, self.weights2) + self.bias2)[0]
        else:
            # Directly compute output
            output = Sigmoid(np.dot(inputs, self.weights) + self.bias)[0]

        # Decision thresholds
        actions = [1 if o > 0.5 else 0 for o in output]
        return actions[:5]  # Return first 5 actions


# Controlle for enemy
class enemy_controller(Controller):
    def __init__(self, _n_hidden):
        self.n_hidden = [_n_hidden]

    def control(self, inputs, controller):
        inputs = (inputs - min(inputs)) / float((max(inputs) - min(inputs)))

        if self.n_hidden[0] > 0:
            bias1 = np.random.rand(1, self.n_hidden[0]) * 0.01
            weights1_slice = len(inputs) * self.n_hidden[0] + self.n_hidden[0]
            weights1 = np.random.randn(len(inputs), self.n_hidden[0]) * np.sqrt(2.0 / len(inputs))
            output1 = ReLu(inputs.dot(weights1) + bias1)

            bias2 = np.random.rand(1, 5) * 0.01
            weights2 = np.random.randn(self.n_hidden[0], 5) * np.sqrt(2.0 / self.n_hidden[0])
            output = Sigmoid(output1.dot(weights2) + bias2)[0]
        else:
            bias = controller[:5].reshape(1, 5)
            weights = controller[5:].reshape((len(inputs), 5))
            output = Sigmoid(inputs.dot(weights) + bias)[0]

        attack1 = 1 if output[0] > 0.5 else 0
        attack2 = 1 if output[1] > 0.5 else 0
        attack3 = 1 if output[2] > 0.5 else 0
        attack4 = 1 if output[3] > 0.5 else 0

        return [attack1, attack2, attack3, attack4]
