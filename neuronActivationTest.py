import numpy as np

# Inputs (17 neurons)
inputs = np.random.randn(17)  # Random inputs for the layer

# Weights (2 x 17 matrix)
weights = np.random.randn(2, 17)  # Random weights for 2 outputs

# Biases (2 neurons)
biases = np.random.randn(2)

# Weighted sum for output layer
z = np.dot(weights, inputs) + biases

# Apply an activation function (e.g., softmax or sigmoid for classification)
output = 1 / (1 + np.exp(-z))  # Sigmoid activation

print("Inputs (17 neurons):", inputs)
print("Weights (2x17):\n", weights)
print("Biases (2):", biases)
print("Pre-activation values (z):", z)
print("Post-activation values (output):", output)
