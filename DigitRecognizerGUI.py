
import tkinter as tk
from tkinter import Canvas
import numpy as np
from PIL import Image, ImageOps

# Define the GUI application
class DigitRecognizerApp:
    def __init__(self, master, predict_function):
        self.master = master
        self.predict_function = predict_function
        self.canvas_size = 560  # Increased canvas size for better visibility
        self.drawing_size = 28  # 28x28 resolution for the network
        self.scale_factor = self.canvas_size // self.drawing_size

        # Canvas for drawing
        self.canvas = Canvas(master, width=self.canvas_size, height=self.canvas_size, bg="white")
        self.canvas.grid(row=0, column=0, columnspan=4)
        self.canvas.bind("<B1-Motion>", self.draw)

        # Buttons
        self.clear_button = tk.Button(master, text="Clear", command=self.clear_canvas)
        self.clear_button.grid(row=1, column=0)

        self.predict_button = tk.Button(master, text="Predict", command=self.predict_digit)
        self.predict_button.grid(row=1, column=3)

        # Label for prediction result
        self.prediction_label = tk.Label(master, text="Draw a digit!", font=("Helvetica", 16))
        self.prediction_label.grid(row=2, column=0, columnspan=4)

        # Initialize drawing
        self.drawing_data = None

    def draw(self, event):
        # Draw scaled circles for smoother drawing
        x, y = event.x, event.y
        self.canvas.create_oval(
            x - self.scale_factor // 2, y - self.scale_factor // 2,
            x + self.scale_factor // 2, y + self.scale_factor // 2,
            fill="black", outline="black"
        )

    def clear_canvas(self):
        self.canvas.delete("all")
        self.prediction_label.config(text="Draw a digit!")

    def predict_digit(self):
        # Convert canvas to image
        canvas_image = self.get_canvas_image()

        # Preprocess the image to match the input format of the network
        processed_image = self.preprocess_image(canvas_image)

        # Get prediction from the neural network
        prediction = self.predict_function(processed_image)

        # Display the prediction
        self.prediction_label.config(text=f"Prediction: {prediction}")

    def get_canvas_image(self):
        # Save the canvas drawing to an image
        self.canvas.postscript(file="canvas.ps", colormode="mono")
        img = Image.open("canvas.ps")
        img = img.convert("L")  # Convert to grayscale
        return img

    def preprocess_image(self, img):
        # Resize to 28x28 without changing resolution
        img = img.resize((self.drawing_size, self.drawing_size), Image.ANTIALIAS)
        img = ImageOps.invert(img)  # Invert colors: white background to black
        img_array = np.array(img)

        # Binarize the image (convert grayscale to 0s and 1s)
        img_array = (img_array > 128).astype(np.float32)

        # Flatten the image to a vector
        return img_array.flatten()

# Mock prediction function for testing
def mock_predict(input_vector):
    # Simulate neural network output probabilities for 10 classes (digits 0-9)
    output_vector = np.random.rand(10)
    predicted_class = np.argmax(output_vector)
    return predicted_class

# Create the GUI and run the app
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Digit Recognizer")
    app = DigitRecognizerApp(root, predict_function=mock_predict)
    root.mainloop()
