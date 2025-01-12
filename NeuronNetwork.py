import numpy as np
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor
from abc import ABC, abstractmethod
import numpy as np
import random

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score


# -------------------------------------------------
# 1. LOSS FUNCTIONS
# -------------------------------------------------

def categorical_cross_entropy_loss(y_true, y_pred, eps=1e-12):
    """
    Vectorized CCE:
    y_true, y_pred: shape (batch_size, num_classes)
    """
    # Clip for numerical stability
    y_pred = np.clip(y_pred, eps, None)
    # Compute average over the batch
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
# -------------------------------------------------
# 2. CUSTOM LAYER (Updated)
# -------------------------------------------------
class Layer:
    def __init__(self, input_dim, output_dim, activation='relu'):
        """
        A single layer in the neural network.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation.lower()

        # He initialization for ReLU, otherwise scaled differently
        if self.activation == 'relu':
            limit = np.sqrt(2.0 / input_dim)
        else:
            limit = np.sqrt(1.0 / input_dim)

        # Initialize weights and biases
        self.weights = np.random.randn(output_dim, input_dim) * limit
        self.bias = np.zeros(output_dim)

        self.z = None       # Pre-activation
        self.output = None  # Post-activation

    def _apply_activation(self, z):
        """Apply the chosen activation function (vectorized for a batch)."""
        if self.activation == 'relu':
            return np.maximum(0, z)
        elif self.activation == 'leaky_relu':
            return np.where(z > 0, z, 0.01 * z)
        elif self.activation == 'sigmoid':
            z = np.clip(z, -500, 500)  # Prevent overflow
            return 1.0 / (1.0 + np.exp(-z))
        elif self.activation == 'tanh':
            z = np.clip(z, -500, 500)  # Prevent overflow
            return np.tanh(z)
        elif self.activation == 'linear':
            return z
        elif self.activation == 'softmax':
            # Stable softmax implementation
            z_clipped = np.clip(z, -100, 100)
            z_shifted = z_clipped - np.max(z_clipped, axis=1, keepdims=True)
            exp_z = np.exp(z_shifted)
            return exp_z / np.sum(exp_z, axis=1, keepdims=True)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def forward_batch(self, X_batch):
        """
        Forward pass: X_batch shape (batch_size, input_dim).
        Returns shape (batch_size, output_dim).
        """
        # Pre-activation (z = Wx + b)
        self.z = X_batch @ self.weights.T + self.bias
        # Apply activation function
        self.output = self._apply_activation(self.z)
        return self.output

    def activation_derivative(self):
        """
        Compute element-wise derivative of the activation function w.r.t. z.
        """
        if self.activation == 'relu':
            return (self.z > 0).astype(float)
        elif self.activation == 'leaky_relu':
            return np.where(self.z > 0, 1.0, 0.01)
        elif self.activation == 'sigmoid':
            return self.output * (1.0 - self.output)
        elif self.activation == 'tanh':
            return 1.0 - self.output ** 2
        elif self.activation == 'linear':
            return np.ones_like(self.z)
        elif self.activation == 'softmax':
            # Softmax gradient is usually handled together with cross-entropy
            # Returning identity for compatibility
            return np.ones_like(self.z)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")


# -------------------------------------------------
# 3. NEURAL NETWORK CORE
# -------------------------------------------------
def categorical_cross_entropy_loss(y_true, y_pred, eps=1e-12):
    """
    y_true, y_pred: shape (batch_size, num_classes)
    """
    y_pred = np.clip(y_pred, eps, None)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))


class NeuronNetwork:
    def __init__(self, layers, learning_rate=0.005):
        self.layers = layers
        self.learning_rate = learning_rate
        self.update_lock = threading.Lock()

    def __getstate__(self):
        state = self.__dict__.copy()
        if 'update_lock' in state:
            del state['update_lock']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.update_lock = threading.Lock()

    def forward_batch(self, X_batch):
        out = X_batch
        for layer in self.layers:
            out = layer.forward_batch(out)
        return out

    def _clip_gradients(self, grads_w, grads_b, clip_value=1.0):
        """
        Clip the gradients to avoid exploding values.
        """
        for i in range(len(grads_w)):
            np.clip(grads_w[i], -clip_value, clip_value, out=grads_w[i])
            np.clip(grads_b[i], -clip_value, clip_value, out=grads_b[i])

    def backward_batch(self, X_batch, y_batch, y_pred):
        """
        Backpropagation for a single batch.
        """
        # Final layer error
        last_layer = self.layers[-1]
        if last_layer.activation == 'softmax':
            delta = (y_pred - y_batch)  # Softmax + cross-entropy
        else:
            error = y_pred - y_batch
            activation_deriv = last_layer.activation_derivative()
            delta = error * activation_deriv

        grads_w = []
        grads_b = []

        # Loop backwards
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            if i == 0:
                prev_output = X_batch
            else:
                prev_output = self.layers[i - 1].output

            grad_w = delta.T @ prev_output
            grad_b = np.sum(delta, axis=0)

            grads_w.append(grad_w)
            grads_b.append(grad_b)

            if i > 0:
                prev_layer = self.layers[i - 1]
                error_next = delta @ layer.weights
                activation_deriv_prev = prev_layer.activation_derivative()

                if prev_layer.activation == 'softmax':
                    delta = error_next
                else:
                    delta = error_next * activation_deriv_prev

        grads_w.reverse()
        grads_b.reverse()

        # --- Gradient Clipping Here ---
        self._clip_gradients(grads_w, grads_b, clip_value=1.0)

        # Update
        with self.update_lock:
            for i, layer in enumerate(self.layers):
                layer.weights -= self.learning_rate * (grads_w[i] / X_batch.shape[0])
                layer.bias_vector -= self.learning_rate * (grads_b[i] / X_batch.shape[0])

    def _train_batch(self, X_batch, y_batch):
        y_pred = self.forward_batch(X_batch)
        batch_loss = categorical_cross_entropy_loss(y_batch, y_pred)
        self.backward_batch(X_batch, y_batch, y_pred)
        return batch_loss

    def train(self,
              X_train,
              y_train,
              epochs=10,
              block_size=64,
              n_threads=2,
              save_interval=0,
              save_path="trained_nn.pkl"):
        n_samples = X_train.shape[0]
        indices = np.arange(n_samples)

        print("Training the neural network...")

        for epoch in range(epochs):
            np.random.shuffle(indices)
            total_loss = 0.0

            with ThreadPoolExecutor(max_workers=n_threads) as executor:
                futures = []
                for start_idx in range(0, n_samples, block_size):
                    end_idx = start_idx + block_size
                    batch_indices = indices[start_idx:end_idx]
                    X_batch = X_train[batch_indices]
                    y_batch = y_train[batch_indices]

                    futures.append(executor.submit(self._train_batch, X_batch, y_batch))

                for f in futures:
                    total_loss += f.result() * block_size

            avg_loss = total_loss / n_samples
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")

            # Save at intervals if requested
            if save_interval > 0 and (epoch + 1) % save_interval == 0:
                self.save(save_path)
                print(f"Model checkpoint saved at epoch {epoch + 1}.")

        # Final save
        self.save(save_path)
        print(f"Training complete. Final model saved to {save_path}.")

    def predict(self, X):
        out = self.forward_batch(X)
        return np.argmax(out, axis=1)

    def feed_single(self, x_single):
        x_batch = x_single.reshape(1, -1)
        output_batch = self.forward_batch(x_batch)
        return output_batch[0]

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)


from abc import ABC, abstractmethod
import numpy as np
import random


class AbstractEnv(ABC):
    @abstractmethod
    def reset(self):
        """
        Reset the environment to the initial state.

        Returns:
        - state (np.ndarray): Initial game state as a vector.
        """
        pass

    @abstractmethod
    def step(self, action):
        """
        Take a step in the environment.

        Parameters:
        - action: The action to take.

        Returns:
        - state (np.ndarray): Current game state as a vector.
        - reward (float): Reward for the step.
        - done (bool): Whether the game is over.
        """
        pass

    @abstractmethod
    def get_state(self):
        """
        Get the current state of the environment as a vector.

        Returns:
        - state (np.ndarray): The current state as a flattened vector.
        """
        pass

    @abstractmethod
    def render(self):
        """
        Render the current state of the environment to the console or a GUI.
        """
        pass

    @abstractmethod
    def get_possible_actions(self):
        """
        Get a list of possible actions the agent can take in the current state.

        Returns:
        - actions (list): List of valid actions.
        """
        pass


class ReinforcerLearner:
    def __init__(self, env: AbstractEnv, layers, learning_rate=0.005):
        """
        Initialize the learner.

        Parameters:
        - env (AbstractEnv): The environment.
        - layers: The layers for the neural network.
        - learning_rate (float): Learning rate for the neural network.
        """
        self.net = NeuronNetwork(layers, learning_rate)
        self.env = env

    def train(self,
              epochs=1,
              steps_per_episode=100,
              discount_factor=0.99,
              save_interval=0,
              save_path="trained_rl_model.pkl",
              do_final_save=True):
        print("Starting policy-gradient training...")

        for epoch_index in range(epochs):
            # Adjust learning rate (optional)
            initial_lr = 0.005
            lr_decay = 0.99
            self.net.learning_rate = initial_lr * (lr_decay ** epoch_index)

            # 1) Reset environment
            self.env.reset()

            # 2) Run episode
            states, actions, rewards = self.run_episode(steps_per_episode)

            # 3) Calculate discounted returns with normalization
            returns = self.get_discounted_returns(rewards, discount_factor)

            # 4) Update policy and calculate loss
            episode_loss = self.update_policy(states, actions, returns)

            # 5) Calculate total return for logging
            episode_return = sum(rewards)

            print(f"Epoch {epoch_index + 1}/{epochs} "
                  f"| Loss: {episode_loss:.4f} "
                  f"| Return: {episode_return:.4f}")

            # 6) Save model at intervals
            if save_interval > 0 and (epoch_index + 1) % save_interval == 0:
                self.net.save(save_path)
                print(f"  [Checkpoint] Model saved at epoch {epoch_index + 1} -> {save_path}")

        # Final save
        if do_final_save:
            self.net.save(save_path)
            print(f"Final model saved to {save_path}.")

    def run_episode(self, steps):
        """
        Run a single episode.

        Parameters:
        - steps (int): Maximum steps in the episode.

        Returns:
        - states (list): List of states observed during the episode.
        - actions (list): List of actions taken during the episode.
        - rewards (list): List of rewards received during the episode.
        """
        states = []
        actions = []
        rewards = []

        for _ in range(steps):
            state = self.env.get_state()
            # Predict probabilities for actions (softmax output)
            probabilities = self.net.feed_single(state)

            # Choose an action probabilistically
            possible_actions = self.env.get_possible_actions()
            action = np.random.choice(possible_actions, p=probabilities)

            # Take the action in the environment
            next_state, reward, done = self.env.step(action)

            # Log state, action, and reward
            states.append(state)
            actions.append(possible_actions.index(action))  # Store index of the action
            rewards.append(reward)

            if done:
                break

        return states, actions, rewards

    def get_discounted_returns(self, rewards, discount_factor):
        """
        Calculate discounted returns.

        Parameters:
        - rewards (list): List of rewards from the episode.
        - discount_factor (float): Discount factor for future rewards.

        Returns:
        - discounted_returns (list): Discounted returns for each time step.
        """
        discounted_returns = []
        return_value = 0
        for reward in reversed(rewards):
            return_value = reward + discount_factor * return_value
            discounted_returns.insert(0, return_value)  # Prepend to maintain correct order
        return discounted_returns

    def update_policy(self, states, actions, returns):
        """
        Update the network using policy gradient.

        Parameters:
        - states (list[np.ndarray]): States from the episode.
        - actions (list[int]): Indices of chosen actions per time step.
        - returns (list[float]): Discounted returns corresponding to each state.

        Returns:
        - float: The total policy loss for this episode (sum of -log(prob) * return).
        """
        # Normalize returns for stability
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-12)

        episode_loss = 0.0

        for state, action, return_value in zip(states, actions, returns):
            # Reshape state to (1, state_dim)
            state_batch = state.reshape(1, -1)

            # Forward pass -> shape (1, action_dim)
            probabilities = self.net.forward_batch(state_batch)

            # Clip to avoid log(0)
            probabilities = np.clip(probabilities, 1e-12, 1.0)

            # The probability of the chosen action
            chosen_action_prob = probabilities[0, action]

            # Policy loss: -log(prob(a|s)) * return
            policy_loss = -np.log(chosen_action_prob) * return_value

            # Entropy bonus: Encourage exploration
            entropy = -np.sum(probabilities * np.log(probabilities))
            total_loss = policy_loss - 0.01 * entropy

            # Accumulate episode loss
            episode_loss += total_loss

            # Backprop with scaled target
            one_hot_action = np.zeros_like(probabilities)
            one_hot_action[0, action] = return_value
            self.net.backward_batch(state_batch, one_hot_action, probabilities)

        return episode_loss


# -------------------------------------------------
# 4. EXAMPLE: TRAIN ON MNIST
# -------------------------------------------------
def main():
    print("Loading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X = mnist.data / 255.0  # Normalize pixel values
    y = mnist.target.astype(np.int32)

    # One-hot encode labels
    ohe = OneHotEncoder(sparse_output=False)
    y_encoded = ohe.fit_transform(y.reshape(-1, 1))

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    print("Data loaded and preprocessed.")

    # Set up layers
    input_dim = X_train.shape[1]   # 784
    hidden_1 = 128
    hidden_2 = 64
    hidden_3 = 32
    output_dim = y_encoded.shape[1]  # 10

    layers = [
        Layer(input_dim, hidden_1, activation='relu'),
        Layer(hidden_1, hidden_2, activation='relu'),
        Layer(hidden_2, hidden_3, activation='relu'),
        Layer(hidden_3, output_dim, activation='softmax')
    ]

    nn = NeuronNetwork(layers=layers, learning_rate=0.005)

    # --- Train using the new self.train(...) method ---
    nn.train(
        X_train,
        y_train,
        epochs=10,        # how many epochs
        block_size=64,    # mini-batch size
        n_threads=4,      # number of threads
        save_interval=2,  # save model every 2 epochs
        save_path="trained_nn_mnist.pkl"
    )

    # Evaluate
    print("Evaluating accuracy on test set...")
    test_preds = nn.predict(X_test)
    true_labels = np.argmax(y_test, axis=1)
    acc = accuracy_score(true_labels, test_preds)
    print(f"Test Set Accuracy: {acc * 100:.2f}%")


if __name__ == "__main__":
    main()