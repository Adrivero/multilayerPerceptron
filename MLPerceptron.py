from threading import activeCount

import numpy as np

class MultilayerPerceptron:
    def __init__(self, input_size, hidden_layers, output_size):
        self.input_dim = input_size
        self.hidden_layers = hidden_layers
        self.output_dim = output_size
        self.total_layers = len(hidden_layers) + 1 #We need to take into account the input layer

        # Initialize random weights and bias vectors
        self.weights = []
        self.biases = []
        layer_sizes = [input_size] + hidden_layers + [output_size]

        for i in range(1, self.total_layers + 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i - 1]))
            self.biases.append(np.random.randn(layer_sizes[i], 1))


    def propagate_forward(self, inputs):
        # Forward propagation
        self.activations = [inputs]
        self.linear_combinations = []

        #Calculates for each layer the activation
        for i in range(self.total_layers):
            linear_output = np.dot(self.weights[i], self.activations[i]) + self.biases[i]
            self.linear_combinations.append(linear_output) #Weight * value of the neuron

            if i < self.total_layers - 1:
                activation = self.activation_function(linear_output)  # Activation for hidden layers
            else:
                activation = linear_output  # Output layer without activation

            self.activations.append(activation)

        return self.activations[-1]  # Final network output


    def propagate_backward(self, inputs, targets):
        examples = inputs.shape[1]  # Number of training samples

        # Stores Weight and biases gradients
        gradients = []

        #Output layer error
        error_signal = self.activations[-1] - targets

        #Loops through the network in inverse order
        for i in range(self.total_layers - 1, -1, -1):
            #Uses chain rule to calculate the derivative of the error respect to weights.
            grad_weights = (1 / examples) * np.dot(error_signal, self.activations[i].T)

            grad_biases = (1 / examples) * np.sum(error_signal, axis=1, keepdims=True)
            gradients.append((grad_weights, grad_biases))


            if i > 0:
                # propagated_error = W^T * error_signal.
                propagated_error = np.dot(self.weights[i].T, error_signal)
                error_signal = propagated_error * self.activation_derivative(self.linear_combinations[i-1])

        return gradients[::-1]  # Reverses the gradients and returns them from input to output


    def update_parameters(self, gradients, step_size):
        # Update weights and biases using gradients
        for i in range(self.total_layers):
            self.weights[i] -= step_size * gradients[i][0]
            self.biases[i] -= step_size * gradients[i][1]


    def activation_function(self, Z):
        # Non-linear activation function
        # Other possible activation functon:
        # return np.tanh(Z)
        return 1 / (1 + np.exp(-Z))


    def activation_derivative(self, Z):
        # Derivative of the activation function
        #Other possible activation functon:
        # return 1 - np.tanh(Z) ** 2
        return self.activation_function(Z)*(1-self.activation_function(Z))