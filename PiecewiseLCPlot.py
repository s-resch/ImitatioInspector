import numpy as np
import matplotlib.pyplot as plt


def piecewise_linear(x):
    if x < 0.5:
        return 0.8697983066164886 * x
    else:
        return 1.176990277991746 * x - 0.176990277991746


# Range of value für x
x_values = np.linspace(0, 1, 500)
y_values = [piecewise_linear(x) for x in x_values]

# Plot
plt.plot(x_values, y_values, label="Piecewise Linear Function")
plt.axvline(0.5, color='blue', linestyle='--', label="Breakingpoint at x=0.5")
plt.axhline(0.5, color='red', linestyle='--', label="f(0.5)=0.5")
plt.axhline(1, color='green', linestyle='--', label="f(1)=1")
plt.axhline(0, color='purple', linestyle='--', label="f(0)=0")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.title("Piecewise Linear Function")
plt.show()
