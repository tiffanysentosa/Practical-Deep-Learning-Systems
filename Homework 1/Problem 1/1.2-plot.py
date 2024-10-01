import numpy as np
import matplotlib.pyplot as plt

# Constants
START = 0
END = 10
SAMPLE_SIZE = 20

# Generate evenly spaced x values
x = np.linspace(start=START, stop=END, num=SAMPLE_SIZE)

# Constants
NOISE_MEAN = 0
NOISE_VARIANCE = 0.3

def f(x):
    return x + np.sin(1.5 * x)

# Generate noise
noise = np.random.normal(loc=NOISE_MEAN, scale=np.sqrt(NOISE_VARIANCE), size=SAMPLE_SIZE)

y = f(x) + noise

plt.scatter(x, y, label='Noisy y(x)', color='blue')
plt.plot(x, f(x), label='f(x) = x + sin(1.5x)', color='red', linestyle='--')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter Plot of y(x) with Noise and f(x) Line')
plt.legend()

plt.show()
