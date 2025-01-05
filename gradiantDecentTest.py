import sympy as sp
import numpy as np
def main():
    # EXAMPLE 1: 2D PLOT
    x, y, z = sp.symbols('x y z')
    f = (sp.sin(x) + sp.sin(y)) +  sp.E ** z

    # Gradient Descent Parameters
    learning_rate = 0.01
    epochs = 1000
    threshold = 1e-6  # Convergence threshold

    # Collect the variables and define the starting point
    variables = list(f.free_symbols)  # [x, y, z]
    starting_input_vector = np.array([1.0, 1.0, 1.0])  # match the number of variables

    # Gradient Descent Loop
    for i in range(epochs):
        gradient_vector = []

        for j in range(len(variables)):
            # Substitute all variables in the gradient calculation
            gradient_value = f.diff(variables[j]).subs({
                variables[k]: starting_input_vector[k] for k in range(len(variables))
            }).evalf()
            gradient_vector.append(float(gradient_value))

        gradient_vector = np.array(gradient_vector)

        # Update the input vector
        starting_input_vector -= gradient_vector * learning_rate

        # Check convergence
        if np.linalg.norm(gradient_vector) < threshold:
            print(f"Converged at epoch {i + 1}, "
                  f"{variables} = {starting_input_vector}, "
                  f"f({variables}) = {f.subs({variables[k]: starting_input_vector[k] for k in range(len(variables))}):.6f}")
            break

        # Print progress
        print(f"Epoch {i + 1}: {variables} = {starting_input_vector}, "
              f"f({starting_input_vector}) = {f.subs({variables[k]: starting_input_vector[k] for k in range(len(variables))}):.6f}, "
              f"gradient = {gradient_vector}")

if __name__ == "__main__":
    main()
