from evoman.controller import Controller
import numpy as np


# ReLU 
def ReLu(x):
    return np.maximum(0, x)
# Sigmoid 
def Sigmoid(x):
    return 1 / (1 + np.exp(-x))


class player_controller(Controller):
    def __init__(self, _n_hidden):
        self.n_hidden = [_n_hidden]

    def set(self, controller, n_inputs):
        if self.n_hidden[0] > 0:
            # He initialization
            # Initialize biases for first hidden layer to small positive values
            self.bias1 = np.random.rand(1, self.n_hidden[0]) * 0.01
            # He initialization for weights for connections from inputs to hidden nodes
            weights1_slice = n_inputs * self.n_hidden[0] + self.n_hidden[0]
            self.weights1 = np.random.randn(n_inputs, self.n_hidden[0]) * np.sqrt(2.0 / n_inputs)

            # Initialize biases output layer
            self.bias2 = np.random.rand(1, 5) * 0.01

            # He initialization for weights for connections from hidden nodes to outputs
            self.weights2 = np.random.randn(self.n_hidden[0], 5) * np.sqrt(2.0 / self.n_hidden[0])

    def control(self, inputs, controller):
        # Normalizes input using min-max scaling
        inputs = (inputs - min(inputs)) / float((max(inputs) - min(inputs)))

        if self.n_hidden[0] > 0:
            # Outputs activation for first hidden layer
            output1 = ReLu(inputs.dot(self.weights1) + self.bias1)

            # Outputs activation for second (output) layer using Sigmoid
            output = Sigmoid(output1.dot(self.weights2) + self.bias2)[0]
        else:
            bias = controller[:5].reshape(1, 5)
            weights = controller[5:].reshape((len(inputs), 5))

            output = Sigmoid(inputs.dot(weights) + bias)[0]

        # Takes decisions about sprite actions based on output threshold (0.5)
        left = 1 if output[0] > 0.5 else 0
        right = 1 if output[1] > 0.5 else 0
        jump = 1 if output[2] > 0.5 else 0
        shoot = 1 if output[3] > 0.5 else 0
        release = 1 if output[4] > 0.5 else 0

        return [left, right, jump, shoot, release]


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
