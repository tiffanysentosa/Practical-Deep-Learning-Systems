import numpy as np
import matplotlib.pyplot as plt

# Constants
START = 0
END = 10
SAMPLE_SIZE = 20
NOISE_MEAN = 0
NOISE_VARIANCE = 0.3

# Generate evenly spaced x values
x = np.linspace(start=START, stop=END, num=SAMPLE_SIZE)

# Generate noise
noise = np.random.normal(loc=NOISE_MEAN, scale=np.sqrt(NOISE_VARIANCE), size=SAMPLE_SIZE)

def f(x):
    return x + np.sin(1.5 * x)

y = f(x) + noise

p1 = np.polyfit(x, y, 1)  #g1(x)
p3 = np.polyfit(x, y, 3)  #g3(x) 
p10 = np.polyfit(x, y, 10) #g10(x) 

# Create the polynomial functions
g1 = np.poly1d(p1)
g3 = np.poly1d(p3)
g10 = np.poly1d(p10)

x_fine = np.linspace(0, 10, 100)


plt.scatter(x, y, label='Noisy y(x)', color='blue')  # Noisy data
plt.plot(x_fine, f(x_fine), label='f(x) = x + sin(1.5x)', color='black', linestyle='--') 
plt.plot(x_fine, g1(x_fine), label='g1(x) - Linear Fit', color='red')  # g1(x)
plt.plot(x_fine, g3(x_fine), label='g3(x) - Cubic Fit', color='green')  # g3(x)
plt.plot(x_fine, g10(x_fine), label='g10(x) - Degree 10 Fit', color='purple')  # g10(x)

plt.xlabel('x')
plt.ylabel('y')
plt.title('Polynomial Estimators of Different Degrees')
plt.legend()

plt.show()
