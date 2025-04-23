import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt

def main():
    # Define a simple function: f(x) = x²
    def f(x):
        return x**2
    
    # Compute gradient function
    df_dx = grad(f)
    
    # Evaluate at a single point
    x = 3.0
    print(f"f({x}) = {f(x)}")
    print(f"f'({x}) = {df_dx(x)}")
    
    # Visualize
    x_range = np.linspace(-5, 5, 100)
    y_values = [f(x_i) for x_i in x_range]
    gradients = [df_dx(x_i) for x_i in x_range]
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x_range, y_values)
    plt.title("f(x) = x²")
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(x_range, gradients)
    plt.title("f'(x) = 2x")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
