import tkinter as tk
import time
import numpy as np
import pickle  # For loading the trained model
from PolicyNetwork import SnakeEnvManhattan,PolicyNetwork  # Replace with your actual Snake environment class

class SnakeGUI:
    def __init__(self, env, policy, cell_size=30, delay=200):
        self.env = env
        self.policy = policy
        self.cell_size = cell_size
        self.delay = delay  # milliseconds between steps

        self.window = tk.Tk()
        self.window.title("Snake AI Simulation")

        self.canvas = tk.Canvas(
            self.window,
            width=self.env.size * self.cell_size,
            height=self.env.size * self.cell_size
        )
        self.canvas.pack()

        self.manual_mode = False  # True if player controls the snake
        self.player_action = None  # Store the player's desired action

        self.bind_keys()
        self.reset()

    def bind_keys(self):
        """
        Bind keyboard keys to control the snake in manual mode.
        """
        self.window.bind("w", lambda event: self.set_player_action("UP"))
        self.window.bind("a", lambda event: self.set_player_action("LEFT"))
        self.window.bind("s", lambda event: self.set_player_action("DOWN"))
        self.window.bind("d", lambda event: self.set_player_action("RIGHT"))
        self.window.bind("m", self.toggle_manual_mode)

    def set_player_action(self, action):
        """
        Update the player's action if in manual mode.
        """
        if self.manual_mode:
            self.player_action = action

    def toggle_manual_mode(self, event=None):
        """
        Toggle between manual and AI control modes.
        """
        self.manual_mode = not self.manual_mode
        self.player_action = None  # Clear any old player input
        print("Manual mode:", "ON" if self.manual_mode else "OFF")

    def reset(self):
        """
        Reset the environment and GUI.
        """
        self.state = self.env.reset()  # Reset the environment
        self.done = False  # Reset the game-over state

        # Clear canvas
        self.canvas.delete("all")

        # Draw grid lines
        for i in range(self.env.size):
            self.canvas.create_line(
                0, i * self.cell_size, self.env.size * self.cell_size, i * self.cell_size
            )
            self.canvas.create_line(
                i * self.cell_size, 0, i * self.cell_size, self.env.size * self.cell_size
            )

        self.render()

    def render(self):
        self.canvas.delete("snake")  # Remove old snake segments
        self.canvas.delete("apple")  # Remove old apple

        # Draw apple
        ar, ac = self.env.apple_pos
        if isinstance(ar, int) and isinstance(ac, int):
            self.canvas.create_rectangle(
                ac * self.cell_size,
                ar * self.cell_size,
                (ac + 1) * self.cell_size,
                (ar + 1) * self.cell_size,
                fill="red",
                tags="apple"
            )
        else:
            print(f"Invalid apple position: {self.env.apple_pos}")

        # Draw snake
        for segment in self.env.snake:
            if isinstance(segment, tuple) and len(segment) == 2:
                sr, sc = segment
                if isinstance(sr, int) and isinstance(sc, int):
                    self.canvas.create_rectangle(
                        sc * self.cell_size,
                        sr * self.cell_size,
                        (sc + 1) * self.cell_size,
                        (sr + 1) * self.cell_size,
                        fill="green",
                        tags="snake"
                    )
                else:
                    print(f"Invalid snake segment: {segment}")
            else:
                print(f"Invalid snake data: {self.env.snake}")

    def step(self):
        """
        Execute a single step in the environment.
        If in manual mode, use the player's input.
        Otherwise, use the policy network.
        """
        if not self.done:
            if self.manual_mode and self.player_action in self.env.actions:
                action = self.player_action
                self.player_action = None  # Clear the action after it's used
            else:
                # Use AI policy
                action_idx, _ = self.policy.choose_action(self.state)
                action = self.env.actions[action_idx]

            self.state, _, self.done = self.env.step(action)
            self.render()

            if self.done:
                print("Game Over!")
                self.window.after(2000, self.reset_game)  # Restart after 2 seconds
            else:
                self.window.after(self.delay, self.step)
                print(self.state)

    def reset_game(self):
        """
        Reset the game after a game-over state.
        """
        self.env.reset()  # Reset the environment
        self.reset()  # Reset the GUI and state
        self.step()  # Start the game loop again

    def start(self):
        """
        Start the simulation.
        """
        self.step()
        self.window.mainloop()


def main():
    # Load the environment
    env = SnakeEnvManhattan(size=16, max_steps=20000)

    # Load the trained policy network
    model_path = r"C:\Users\etay1\PycharmProjects\pythonProject\mathyStuff\snake_models\policy_net_1000000.pkl"
    with open(model_path, "rb") as f:
        policy_net = pickle.load(f)

    # Start the GUI
    gui = SnakeGUI(env, policy_net, cell_size=50, delay=100)
    gui.start()


if __name__ == "__main__":
    main()
