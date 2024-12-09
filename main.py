import numpy as np
from MLPerceptron import MultilayerPerceptron
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

#-------------------------------------------------------------------------------------------------------------
# Expected value: 0.5 something. Could get various executions to get it.
# This low test value clearly indicates that overfitting is occuring as the model has adjusted too much to our test set
# If we change the activating function to tanh, trained test loss goes to 1.73. More acceptable.
#-------------------------------------------------------------------------------------------------------------


# Creates a dummy regression dataset
data_inputs, data_targets = make_regression(n_samples=100, n_features=1, noise=0.1, random_state=42)

# Split data into the training and testing set
inputs_train, inputs_test, targets_train, targets_test = train_test_split(
    data_inputs, data_targets, test_size=0.2, random_state=42)


# Normalize the input data
inputs_mean = np.mean(inputs_train)
inputs_std = np.std(inputs_train)
inputs_train = (inputs_train - inputs_mean) / inputs_std
inputs_test = (inputs_test - inputs_mean) / inputs_std

# Makes targets as column vectors
targets_train = targets_train.reshape(-1, 1)
targets_test = targets_test.reshape(-1, 1)


# Define the neural network
input_dim = inputs_train.shape[1]
hidden_layers = [10, 10]
output_dim = targets_train.shape[1]
nn = MultilayerPerceptron(input_dim, hidden_layers, output_dim)

# Training parameters
epochs = 1000
learning_rate = 0.01

# Training loop
for epoch in range(epochs):
    # Forward propagation
    predictions = nn.propagate_forward(inputs_train.T)

    # Backward propagation and parameter update
    gradients = nn.propagate_backward(inputs_train.T, targets_train.T)
    nn.update_parameters(gradients, learning_rate)

    # Calculate and display the loss
    loss = np.mean((predictions - targets_train.T) ** 2)
    if (epoch + 1) % 100 == 0:
        print(f"Iteration: {epoch + 1} - Loss: {loss}")

# Final testing on the test set
test_predictions = nn.propagate_forward(inputs_test.T)
test_loss = np.mean((test_predictions - targets_test.T) ** 2)
print(f"Trained Test Loss: {test_loss}")

