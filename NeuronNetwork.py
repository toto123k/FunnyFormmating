import numpy as np
import random

# -------------------------------------------------
# 1. LOSS FUNCTIONS
# -------------------------------------------------

def cross_entropy_loss(y_true, y_pred):
    """
    Calculate cross-entropy loss for a single-output model (0 <= y_pred <= 1).
    """
    y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def mse_loss(y_true, y_pred):
    """
    Mean Squared Error (MSE) loss.
    """
    return np.mean((y_true - y_pred)**2)

def mae_loss(y_true, y_pred):
    """
    Mean Absolute Error (MAE) loss.
    """
    return np.mean(np.abs(y_true - y_pred))


# -------------------------------------------------
# 2. SINGLE LAYER
# -------------------------------------------------
class Layer:
    def __init__(self, input_dim, output_dim, activation='relu'):
        """
        A single layer in the neural network.

        Parameters:
        - input_dim  (int): Number of inputs to this layer
        - output_dim (int): Number of neurons (outputs) in this layer
        - activation (str): Activation function ('relu', 'sigmoid', 'tanh', 'linear')
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation.lower()

        # Initialize weights and biases
        self.weights = np.random.randn(output_dim, input_dim)
        self.bias_vector = np.random.randn(output_dim)

        # Storage for forward pass
        self.pre_activation = None
        self.output = None

    def _apply_activation(self, z):
        """Apply the chosen activation function."""
        if self.activation == 'relu':
            return np.maximum(0, z)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        elif self.activation == 'tanh':
            return np.tanh(z)
        elif self.activation == 'linear':
            return z
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def apply_activation_derivative(self):
        """
        Calculate the derivative of the activation function.
        """
        if self.activation == 'relu':
            return np.where(self.pre_activation > 0, 1, 0)
        elif self.activation == 'sigmoid':
            return self.output * (1 - self.output)
        elif self.activation == 'tanh':
            return 1 - self.output**2
        elif self.activation == 'linear':
            return np.ones_like(self.pre_activation)
        else:
            raise ValueError(f"Unknown activation function: {self.activation}")

    def feed(self, input_vector):
        """
        Forward pass through this layer:
            pre_activation = W * x + b
            output = activation(pre_activation)
        """
        self.pre_activation = self.weights.dot(input_vector) + self.bias_vector
        self.output = self._apply_activation(self.pre_activation)
        return self.output


# -------------------------------------------------
# 3. NEURAL NETWORK
# -------------------------------------------------
class NeuronNetwork:
    def __init__(self, layers, learning_rate=0.01, loss_type='cross_entropy'):
        """
        A flexible neural network that accepts manually created Layer objects.

        Parameters:
        - layers (list[Layer]): The layer objects for the network architecture
        - learning_rate (float): Learning rate for gradient descent
        - loss_type (str)      : 'cross_entropy', 'mse', or 'mae'
        """
        self.layers = layers
        self.learning_rate = learning_rate

        if loss_type == 'cross_entropy':
            self.loss_fn = cross_entropy_loss
        elif loss_type == 'mse':
            self.loss_fn = mse_loss
        elif loss_type == 'mae':
            self.loss_fn = mae_loss
        else:
            raise ValueError("loss_type must be 'cross_entropy', 'mse', or 'mae'.")
        self.loss_type = loss_type

    def forward_propagate(self, input_vector):
        """Perform a forward pass through the entire network."""
        for layer in self.layers:
            input_vector = layer.feed(input_vector)
        return input_vector

    def fit_single(self, input_vector, actual_output):
        """
        Train on a single data point with backpropagation.

        Parameters:
        - input_vector  (np.array): The input features
        - actual_output (float)   : The target label in [0,1]
        """
        # 1. Forward pass
        output = self.forward_propagate(input_vector)

        # 2. Error
        error = output - actual_output

        # 3. Backpropagation
        for layer_index in reversed(range(len(self.layers))):
            layer = self.layers[layer_index]

            # delta = error * derivative_of_activation
            delta = error * layer.apply_activation_derivative()

            # Update weights and biases
            if layer_index == 0:
                prev_output = input_vector
            else:
                prev_output = self.layers[layer_index - 1].output

            layer.weights -= self.learning_rate * np.outer(delta, prev_output)
            layer.bias_vector -= self.learning_rate * delta

            # Propagate error to the previous layer
            if layer_index > 0:
                error = layer.weights.T @ delta

        return self.loss_fn(actual_output, output)

    def train(self, inputs, outputs, epochs=1000, print_interval=100):
        """
        Train the network for multiple epochs.

        Parameters:
        - inputs (list[np.array]): The input data
        - outputs (list[float])  : The target values
        - epochs (int)           : Number of training epochs
        - print_interval (int)   : Print loss every 'print_interval' epochs
        """
        for epoch in range(epochs):
            total_loss = 0
            for x, y in zip(inputs, outputs):
                total_loss += self.fit_single(x, y)
            if epoch % print_interval == 0:
                avg_loss = total_loss / len(inputs)
                print(f"Epoch {epoch}, {self.loss_type} loss: {avg_loss:.6f}")

    def pred(self, input_vector):
        """Forward pass for a single input to get a prediction."""
        return self.forward_propagate(input_vector)

