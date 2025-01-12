import os
import pickle
import numpy as np
import random
from typing import List

# Use your `Layer` and `NeuronNetwork` classes directly
from NeuronNetwork import Layer, NeuronNetwork  # Replace with your actual path to the file


# ------------------------------------------------------
# 3) POLICY NETWORK (Updated with Training & Saving)
# ------------------------------------------------------
class PolicyNetwork:
    def __init__(self, state_dim, hidden1=64, hidden2=32,
                 action_dim=4, lr=1e-3, gamma=0.99):
        self.gamma = gamma
        self.lr = lr

        # Build layers using your custom Layer implementation
        self.layer1 = Layer(state_dim, hidden1, activation='relu')
        self.layer2 = Layer(hidden1, hidden2, activation='relu')
        self.layer3 = Layer(hidden2, action_dim, activation='softmax')
        self.layers = [self.layer1, self.layer2, self.layer3]

    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward_batch(out)  # Use forward_batch for your Layer
        return out  # shape (batch_size, action_dim)

    def choose_action(self, state):
        """
        Given a single state (shape=(state_dim,)),
        return (action_idx, prob_of_chosen_action).
        """
        X = state.reshape(1, -1)
        probs = self.forward(X)[0]  # shape=(action_dim,)
        action_idx = np.random.choice(len(probs), p=probs)
        return action_idx, probs[action_idx]

    def discount_rewards(self, rewards):
        """
        Compute discounted returns from a list of rewards.
        """
        discounted = np.zeros_like(rewards, dtype=np.float32)
        running_sum = 0
        for i in reversed(range(len(rewards))):
            running_sum = rewards[i] + self.gamma * running_sum
            discounted[i] = running_sum
        return discounted

    def reinforce_update(self, states, actions, discounted_returns):
        """
        Perform a single REINFORCE update after one episode.
        states: shape (N, state_dim)
        actions: shape (N,)
        discounted_returns: shape (N,)
        """
        # 1) Forward pass: get policy distribution
        out = self.forward(states)  # shape = (N, action_dim)

        # Build "one-hot" for chosen actions:
        N, action_dim = out.shape
        one_hot = np.zeros_like(out)
        for i, act in enumerate(actions):
            one_hot[i, act] = 1.0

        # Error = (out - one_hot) * discounted_returns
        # Each row is scaled by the return
        error = (out - one_hot) * discounted_returns.reshape(-1, 1)

        # --- BACKPROP ---

        # LAYER 3
        layer3 = self.layers[-1]
        prev_out = self.layers[-2].output  # shape=(N, hidden2)
        grad_w3 = error.T @ prev_out       # shape=(action_dim, hidden2)
        grad_b3 = np.sum(error, axis=0)    # shape=(action_dim,)

        # For a pure softmax output layer, we typically skip derivative
        deriv3 = layer3.activation_derivative()  # Should be None for softmax
        if deriv3 is not None:
            error *= deriv3
        delta = error @ layer3.weights      # shape=(N, hidden2)

        # LAYER 2
        layer2 = self.layers[-2]
        prev_out = self.layers[-3].output  # shape=(N, hidden1)
        deriv2 = layer2.activation_derivative()  # shape=(N, hidden2)
        delta *= deriv2
        grad_w2 = delta.T @ prev_out       # shape=(hidden2, hidden1)
        grad_b2 = np.sum(delta, axis=0)    # shape=(hidden2,)

        # Propagate delta
        delta = delta @ layer2.weights     # shape=(N, hidden1)

        # LAYER 1
        layer1 = self.layers[-3]
        prev_out = states                  # shape=(N, state_dim)
        deriv1 = layer1.activation_derivative()  # shape=(N, hidden1)
        delta *= deriv1
        grad_w1 = delta.T @ prev_out       # shape=(hidden1, state_dim)
        grad_b1 = np.sum(delta, axis=0)    # shape=(hidden1,)

        # Update weights (gradient descent)
        layer1.weights -= (self.lr / N) * grad_w1
        layer1.bias -= (self.lr / N) * grad_b1
        layer2.weights -= (self.lr / N) * grad_w2
        layer2.bias -= (self.lr / N) * grad_b2
        layer3.weights -= (self.lr / N) * grad_w3
        layer3.bias -= (self.lr / N) * grad_b3

    def train_one_episode(self, env, max_steps=50):
        """
        1) Run an episode in env, collect (state, action, reward)
        2) Compute discounted returns
        3) Single REINFORCE update
        """
        states = []
        actions = []
        rewards = []

        state = env.reset()
        total_reward = 0.0

        for _ in range(max_steps):
            action_idx, _ = self.choose_action(state)
            next_state, reward, done = env.step(env.actions[action_idx])

            states.append(state)
            actions.append(action_idx)
            rewards.append(reward)
            total_reward += reward

            state = next_state
            if done:
                break

        # Convert to numpy
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=int)
        rewards = np.array(rewards, dtype=np.float32)

        # Discount
        discounted_returns = self.discount_rewards(rewards)

        # Normalize returns (optional, helps with stability)
        discounted_returns = (discounted_returns - discounted_returns.mean()) / \
                             (discounted_returns.std() + 1e-8)

        # Update
        self.reinforce_update(states, actions, discounted_returns)
        return total_reward

    def train(self, env, num_episodes=1000, max_steps=50,
              print_interval=50, save_interval=200,
              save_dir='trained_models'):
        """
        Train the policy on the given environment for `num_episodes`.
        - Prints average return every `print_interval` episodes.
        - Saves a checkpoint every `save_interval` episodes into `save_dir`.
        """
        os.makedirs(save_dir, exist_ok=True)  # Create folder if doesn't exist

        returns = []
        for episode in range(num_episodes):
            ep_return = self.train_one_episode(env, max_steps=max_steps)
            returns.append(ep_return)

            # Print progress
            if (episode + 1) % print_interval == 0:
                avg_ret = np.mean(returns[-print_interval:])
                print(f"Episode {episode+1}/{num_episodes} | "
                      f"Avg Return (last {print_interval}): {avg_ret:.3f}")

            # Save checkpoint
            if (episode + 1) % save_interval == 0:
                filename = os.path.join(save_dir, f"policy_net_{episode+1}.pkl")
                with open(filename, 'wb') as f:
                    pickle.dump(self, f)

        print("Training complete.")

    def test(self, env, max_steps=200):
        """
        Test the policy on the given environment for `max_steps`.
        Returns the total reward obtained.
        """
        state = env.reset()
        done = False
        test_ret = 0.0
        steps = 0
        while not done and steps < max_steps:
            action_idx, _ = self.choose_action(state)
            state, reward, done = env.step(env.actions[action_idx])
            test_ret += reward
            steps += 1
        return test_ret

# ------------------------------------------------------
# 1) SIMPLE 4x4 GRID ENV (as before)
# ------------------------------------------------------
class SimpleGridEnv:
    def __init__(self, size=4, max_steps=50):
        self.size = size
        self.max_steps = max_steps
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        self.reset()

    def reset(self):
        self.agent_row = 0
        self.agent_col = 0
        self.steps_taken = 0
        return self.get_state()

    def get_state(self):
        return np.array([
            self.agent_row / (self.size - 1),
            self.agent_col / (self.size - 1)
        ], dtype=np.float32)

    def step(self, action):
        self.steps_taken += 1
        if action == 'UP':
            self.agent_row = max(0, self.agent_row - 1)
        elif action == 'DOWN':
            self.agent_row = min(self.size - 1, self.agent_row + 1)
        elif action == 'LEFT':
            self.agent_col = max(0, self.agent_col - 1)
        elif action == 'RIGHT':
            self.agent_col = min(self.size - 1, self.agent_col + 1)

        reward = -0.01
        done = False
        if self.agent_row == self.size - 1 and self.agent_col == self.size - 1:
            reward = 1.0
            done = True

        if self.steps_taken >= self.max_steps:
            done = True

        return self.get_state(), reward, done

    def get_possible_actions(self):
        return self.actions

class SnakeEnvManhattan:
    def __init__(self, size=6, max_steps=200):
        self.size = size
        self.max_steps = max_steps
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        self.reset()

    def reset(self):
        self.done = False
        self.steps_taken = 0

        # Start snake in the middle
        mid = self.size // 2
        self.snake = [(mid, mid)]  # list of (row, col) (head is snake[0])
        self.snake_direction = 'UP'  # initial direction
        self._place_apple()

        return self.get_state()

    def _place_apple(self):
        while True:
            row = random.randint(0, self.size - 1)
            col = random.randint(0, self.size - 1)
            if (row, col) not in self.snake:
                self.apple_pos = (row, col)
                break

    def get_state(self):
        """
        Returns a feature vector:
         - Boolean one-hot encoded for whether the next step in each direction (UP, DOWN, LEFT, RIGHT)
           will hit the tail or wall.
         - Normalized x and y distance to apple.
        """
        head_r, head_c = self.snake[0]

        # Boolean features for collisions
        will_hit_tail_up = (head_r - 1, head_c) in self.snake or head_r - 1 < 0
        will_hit_tail_down = (head_r + 1, head_c) in self.snake or head_r + 1 >= self.size
        will_hit_tail_left = (head_r, head_c - 1) in self.snake or head_c - 1 < 0
        will_hit_tail_right = (head_r, head_c + 1) in self.snake or head_c + 1 >= self.size

        # Normalized x and y distance to apple
        apple_r, apple_c = self.apple_pos
        x_distance_norm = (apple_c - head_c) / (self.size - 1)
        y_distance_norm = (apple_r - head_r) / (self.size - 1)

        return np.array([
            int(will_hit_tail_up),
            int(will_hit_tail_down),
            int(will_hit_tail_left),
            int(will_hit_tail_right),
            x_distance_norm,
            y_distance_norm
        ], dtype=np.float32)

    def step(self, action):
        if self.done:
            return self.get_state(), 0.0, True

        self.steps_taken += 1
        self.snake_direction = action

        head_r, head_c = self.snake[0]
        if self.snake_direction == 'UP':
            head_r -= 1
        elif self.snake_direction == 'DOWN':
            head_r += 1
        elif self.snake_direction == 'LEFT':
            head_c -= 1
        elif self.snake_direction == 'RIGHT':
            head_c += 1

        # Check wall collision
        if head_r < 0 or head_r >= self.size or head_c < 0 or head_c >= self.size:
            self.done = True
            return self.get_state(), -10.0, True

        # Check body collision
        if (head_r, head_c) in self.snake:
            self.done = True
            return self.get_state(), -20.0, True  # Strong punishment for hitting the tail

        new_head = (head_r, head_c)
        self.snake.insert(0, new_head)

        reward = 0.1  # Reward for staying alive

        # Check if the snake eats the apple
        if new_head == self.apple_pos:
            reward = 1.0
            self._place_apple()
        else:
            self.snake.pop()

        # Check if agent moves closer to the apple
        # Handle the case where the snake has only one segment
        if len(self.snake) > 1:
            x_distance_before = abs(self.snake[1][1] - self.apple_pos[1])
            y_distance_before = abs(self.snake[1][0] - self.apple_pos[0])
        else:
            x_distance_before = abs(self.snake[0][1] - self.apple_pos[1])
            y_distance_before = abs(self.snake[0][0] - self.apple_pos[0])

        x_distance_after = abs(head_c - self.apple_pos[1])
        y_distance_after = abs(head_r - self.apple_pos[0])

        if x_distance_after + y_distance_after < x_distance_before + y_distance_before:
            reward += 0.5  # Reward for moving closer to the apple

        # Check max steps
        if self.steps_taken >= self.max_steps:
            self.done = True

        return self.get_state(), reward, self.done

    def get_possible_actions(self):
        return self.actions



def main():
    env = SnakeEnvManhattan(size=16, max_steps=2000)

    # Update state_dim for the new state representation
    state_dim = len(env.get_state())  # 6 features: 4 for collisions + 2 for x/y distances
    action_dim = len(env.actions)

    model_path = r''

    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            policy_net = pickle.load(f)
        print(f"Loaded pre-trained policy network from {model_path}")
    else:
        policy_net = PolicyNetwork(
            state_dim=state_dim,
            hidden1=128,
            hidden2=64,
            action_dim=action_dim,
            lr=1e-3,
            gamma=0.99
        )
        print("Initialized a new policy network.")

    policy_net.train(
        env=env,
        num_episodes=1000000,
        max_steps=100,
        print_interval=100,
        save_interval=10000,
        save_dir='snake_models'
    )

    test_return = policy_net.test(env, max_steps=2000)
    print(f"\n[Testing final policy] Return: {test_return:.3f}")

    with open("trained_rl_model_final.pkl", 'wb') as f:
        pickle.dump(policy_net, f)
    print("Final model saved to trained_rl_model_final.pkl")


if __name__ == "__main__":
    main()
