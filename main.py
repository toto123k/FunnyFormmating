import numpy as np
import random

from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ---------------------------------------------------------------------------
# 1. Our Custom Neural Network Classes (as from your codebase NeuronNetwork.py)
# ---------------------------------------------------------------------------
def cross_entropy_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def mae_loss(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

class Layer:
    def __init__(self, input_dim, output_dim, activation='relu'):
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
        self.pre_activation = self.weights.dot(input_vector) + self.bias_vector
        self.output = self._apply_activation(self.pre_activation)
        return self.output

class NeuronNetwork:
    def __init__(self, layers, learning_rate=0.01, loss_type='mse'):
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
        for layer in self.layers:
            input_vector = layer.feed(input_vector)
        return input_vector

    def fit_single(self, input_vector, actual_output):
        # Forward pass
        output = self.forward_propagate(input_vector)

        # Error
        error = output - actual_output

        # Backprop
        for layer_index in reversed(range(len(self.layers))):
            layer = self.layers[layer_index]
            delta = error * layer.apply_activation_derivative()

            # Update weights
            if layer_index == 0:
                prev_output = input_vector
            else:
                prev_output = self.layers[layer_index - 1].output

            layer.weights -= self.learning_rate * np.outer(delta, prev_output)
            layer.bias_vector -= self.learning_rate * delta

            if layer_index > 0:
                error = layer.weights.T @ delta

        return self.loss_fn(actual_output, output)

    def train(self, inputs, outputs, epochs=1000, print_interval=100):
        for epoch in range(epochs):
            total_loss = 0
            for x, y in zip(inputs, outputs):
                total_loss += self.fit_single(x, y)
            if epoch % print_interval == 0:
                avg_loss = total_loss / len(inputs)
                print(f"Epoch {epoch}, {self.loss_type} loss: {avg_loss:.6f}")

    def pred(self, input_vector):
        return self.forward_propagate(input_vector)

# ---------------------------------------------------------------------------
# 2. Helper Functions for One-Hot Encoding & Data Generation
# ---------------------------------------------------------------------------
def one_hot_eye_color(eye_color):
    vec = np.zeros(3)
    idx = min(max(eye_color, 0), 2)
    vec[idx] = 1
    return vec

def one_hot_hair_color(hair_color):
    vec = np.zeros(4)
    idx = min(max(hair_color - 1, 0), 3)
    vec[idx] = 1
    return vec

def scale_height(h):
    return (h - 1.5) / 0.5  # map [1.5, 2.0] to [0, 1]

def generate_one_hot_sample(height, eye_c, hair_c, scale=True):
    h_val = scale_height(height) if scale else height
    e_vec = one_hot_eye_color(eye_c)
    h_vec = one_hot_hair_color(hair_c)
    return np.concatenate(([h_val], e_vec, h_vec))

def generate_fake_dataset_one_hot(size=30, scale=True):
    inputs = []
    outputs = []
    for _ in range(size):
        height = 1.5 + random.random() * 0.5
        eye_color = random.choice([0,1,2])
        hair_color = random.choice([1,2,3,4])

        # "secret formula" for attractiveness
        score = 0.0
        score += (height - 1.5) * 1.2
        if eye_color == 1:
            score += 0.1
        elif eye_color == 2:
            score += 0.05
        if hair_color == 2:
            score += 0.1
        elif hair_color == 3:
            score += 0.2
        elif hair_color == 4:
            score += 0.15
        else:
            score += 0.05

        # clamp
        score = max(0, min(score, 1))

        x_vec = generate_one_hot_sample(height, eye_color, hair_color, scale=scale)
        inputs.append(x_vec)
        outputs.append(score)

    return np.array(inputs), np.array(outputs)

# ---------------------------------------------------------------------------
# 3. MAIN - Compare with Random Forest & KNN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Generate data
    X, y = generate_fake_dataset_one_hot(size=1000, scale=True)

    # Split data into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # 3.1 Train Our Custom Neural Network
    layer1 = Layer(8, 6, 'relu')    
    layer2 = Layer(6, 3, 'relu')    
    layer3 = Layer(3, 1, 'sigmoid') 
    net = NeuronNetwork(
        layers=[layer1, layer2, layer3],
        learning_rate=0.05,
        loss_type='mse'
    )

    print("Training Custom Neural Network...\n")
    net.train(
        inputs=X_train,
        outputs=y_train,
        epochs=1000,
        print_interval=500
    )

    # Evaluate custom NN on the test set
    y_pred_nn = np.array([net.pred(x)[0] for x in X_test])  
    mse_nn = mean_squared_error(y_test, y_pred_nn)

    print(f"\nCustom NN Test MSE = {mse_nn:.6f}")

    # 3.2 Random Forest Regressor
    print("\nTraining Random Forest Regressor...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    print(f"Random Forest Test MSE = {mse_rf:.6f}")

    # 3.3 K-Nearest Neighbors Regressor
    print("\nTraining KNN Regressor...")
    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    mse_knn = mean_squared_error(y_test, y_pred_knn)
    print(f"KNN Test MSE = {mse_knn:.6f}")

    # Show final comparison
    print("\n---- Final Comparison (Test MSE) ----")
    print(f"Custom Neural Network: {mse_nn:.6f}")
    print(f"Random Forest       : {mse_rf:.6f}")
    print(f"KNN                 : {mse_knn:.6f}")
