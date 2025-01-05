
class FunctionPlotter:
    def __init__(self, function, x_range=(-2 * np.pi, 2 * np.pi), y_range=(-2 * np.pi, 2 * np.pi),
                 num_points=100, dim=2):
        """
        Initialize the FunctionPlotter class.

        Parameters:
            function : A sympy symbolic function.
                       - If it has 1 free symbol, we do 2D plotting.
                       - If it has 2 free symbols, we do 3D plotting.
            x_range : Tuple specifying the range of x values (for 2D or 3D).
            y_range : Tuple specifying the range of y values (for 3D).
            num_points : Number of points in each dimension to plot the function.
            dim : Dimension of plot. 2 for 2D, 3 for 3D.
        """
        self.dim = dim
        self.function = function

        # Extract the free symbols
        self.variables = list(function.free_symbols)
        self.variables.sort(key=lambda v: v.name)  # Sort to have a predictable order

        if dim == 2:
            # Expect 1-variable function
            self.x = self.variables[0]
            self.f_numeric = sp.lambdify(self.x, self.function, modules=["numpy"])
            self.x_vals = np.linspace(x_range[0], x_range[1], num_points)
            self.y_vals = self.f_numeric(self.x_vals)

            # Prepare the figure and axis for 2D
            self.figure, self.ax = plt.subplots()
            self.plot_function_2d()

        elif dim == 3:
            # Expect 2-variable function
            if len(self.variables) != 2:
                raise ValueError(
                    "For 3D plots, the function must have exactly 2 variables."
                )
            self.x, self.y = self.variables
            self.f_numeric = sp.lambdify((self.x, self.y), self.function, modules=["numpy"])

            # Generate data in two dimensions
            self.x_vals = np.linspace(x_range[0], x_range[1], num_points)
            self.y_vals = np.linspace(y_range[0], y_range[1], num_points)
            self.X, self.Y = np.meshgrid(self.x_vals, self.y_vals)
            self.Z = self.f_numeric(self.X, self.Y)

            # Prepare the figure and axis for 3D
            self.figure = plt.figure()
            self.ax = self.figure.add_subplot(111, projection='3d')
            self.plot_function_3d()

        else:
            raise ValueError("dim must be 2 or 3.")

        # For point updates
        self.point = None
        self.point_plot = None

        plt.ion()  # Enable interactive mode
        plt.show()

    def plot_function_2d(self):
        """Plots the 1D function (2D graph) once."""
        self.ax.plot(self.x_vals, self.y_vals, label=str(self.function), color='blue')
        self.ax.axhline(0, color='black', linewidth=0.8, linestyle="--")
        self.ax.axvline(0, color='black', linewidth=0.8, linestyle="--")
        self.ax.set_title("2D Function Plot")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("f(x)")
        self.ax.grid(True)
        self.ax.legend()

    def plot_function_3d(self):
        """Plots the 2D function (3D surface) once."""
        self.ax.plot_surface(self.X, self.Y, self.Z, cmap='viridis', edgecolor='none', alpha=0.8)
        self.ax.set_title("3D Function Plot")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("f(x, y)")

    def update_point(self, x_value, y_value=None):
        """
        Updates the point on the function.

        Parameters:
            x_value : The x-coordinate of the point (for 2D or 3D).
            y_value : The y-coordinate of the point (only relevant for 3D).
        """
        if self.dim == 2:
            # Calculate the corresponding y-value for f(x)
            y_val = self.f_numeric(x_value)
            self.point = (x_value, y_val)

            # Update the point without clearing the function
            if self.point_plot:
                self.point_plot.remove()
            self.point_plot = self.ax.scatter(
                [x_value], [y_val], color='red',
                label=f'({x_value:.2f}, {y_val:.2f})'
            )
            self.ax.legend()
            self.figure.canvas.draw()
            self.figure.canvas.flush_events()

        elif self.dim == 3:
            if y_value is None:
                raise ValueError("Must provide both x_value and y_value for 3D plots.")

            # Evaluate Z = f(x, y)
            z_val = self.f_numeric(x_value, y_value)
            self.point = (x_value, y_value, z_val)

            # Update the point in 3D
            if self.point_plot:
                self.point_plot.remove()
            self.point_plot = self.ax.scatter(
                [x_value], [y_value], [z_val],
                color='red', s=50,
                label=f'({x_value:.2f}, {y_value:.2f}, {z_val:.2f})'
            )
            self.ax.legend()
            self.figure.canvas.draw()
            self.figure.canvas.flush_events()

        else:
            raise ValueError("dim must be 2 or 3.")
