import tkinter as tk
from tkinter import messagebox
import numpy as np
from PIL import Image, ImageOps, ImageGrab
import pickle
from scipy.ndimage import center_of_mass
from scipy.ndimage.interpolation import shift
from NeuronNetwork import  *
# -------------------------------------------------
# Load the trained model
# -------------------------------------------------
model_file = "trained_nn_mnist.pkl"
with open(model_file, 'rb') as f:
    nn = pickle.load(f)

# -------------------------------------------------
# Helper Functions
# -------------------------------------------------

def center_image(image_array):
    """
    Center the digit in the 28x28 grid by shifting the center of mass to the center.

    Args:
        image_array (np.array): Array of shape (28, 28).

    Returns:
        np.array: Centered image array.
    """
    cy, cx = center_of_mass(image_array)
    shift_y = 14 - int(cy)  # 14 is the center of a 28x28 image
    shift_x = 14 - int(cx)
    return shift(image_array, shift=(shift_y, shift_x), mode='constant')

def preprocess_canvas_data(canvas_image):
    """
    Process the canvas data to match the MNIST format.

    Args:
        canvas_image (PIL.Image): Image object from the canvas.

    Returns:
        np.array: Flattened array of shape (1, 784) for prediction.
    """
    # Ensure the image is grayscale
    img = canvas_image.convert("L")

    # Resize to 28x28 pixels (MNIST format)
    img = img.resize((28, 28), Image.Resampling.LANCZOS)

    # Normalize pixel values to [0, 1]
    img_array = np.array(img) / 255.0

    # Invert colors (MNIST digits are white on black background)
    img_array = 1 - img_array

    # Center the digit
    img_array = center_image(img_array)

    # Save for debugging (optional)
    Image.fromarray((img_array * 255).astype('uint8')).save("preprocessed_digit.png")

    # Flatten the image
    return img_array.flatten().reshape(1, -1)

def predict_digit(canvas_image):
    """
    Predict the digit using the trained neural network.

    Args:
        canvas_image (PIL.Image): Image object from the canvas.

    Returns:
        int: Predicted digit
    """
    processed_data = preprocess_canvas_data(canvas_image)
    prediction = nn.predict(processed_data)
    return prediction[0]

# -------------------------------------------------
# GUI Implementation
# -------------------------------------------------

class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Recognizer")

        # Canvas for drawing
        self.canvas = tk.Canvas(root, width=280, height=280, bg="white")
        self.canvas.grid(row=0, column=0, padx=10, pady=10)

        # Buttons
        self.predict_button = tk.Button(root, text="Predict", command=self.predict)
        self.predict_button.grid(row=1, column=0, pady=5)

        self.clear_button = tk.Button(root, text="Clear", command=self.clear_canvas)
        self.clear_button.grid(row=2, column=0, pady=5)

        # Drawing state
        self.drawing = False
        self.last_x, self.last_y = None, None

        # Bind mouse events for drawing
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)

    def start_draw(self, event):
        self.drawing = True
        self.last_x, self.last_y = event.x, event.y

    def draw(self, event):
        if self.drawing:
            x, y = event.x, event.y
            self.canvas.create_line((self.last_x, self.last_y, x, y), width=10, fill="black", capstyle=tk.ROUND, smooth=True)
            self.last_x, self.last_y = x, y

    def stop_draw(self, event):
        self.drawing = False

    def clear_canvas(self):
        self.canvas.delete("all")

    def predict(self):
        # Get the canvas content as an image
        canvas_image = self.get_canvas_image()

        # Predict the digit
        predicted_digit = predict_digit(canvas_image)

        # Show result
        messagebox.showinfo("Prediction", f"Predicted Digit: {predicted_digit}")

    def get_canvas_image(self):
        """
        Capture the current state of the canvas as a PIL Image.

        Returns:
            PIL.Image: Image object of the canvas
        """
        # Get the canvas coordinates
        x = self.canvas.winfo_rootx()
        y = self.canvas.winfo_rooty()
        w = x + self.canvas.winfo_width()
        h = y + self.canvas.winfo_height()

        # Capture the canvas area from the screen
        canvas_image = ImageGrab.grab(bbox=(x, y, w, h))

        # Convert to grayscale
        canvas_image = canvas_image.convert("L")

        return canvas_image

# -------------------------------------------------
# Run the Application
# -------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()
