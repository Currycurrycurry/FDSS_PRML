import numpy as np
import matplotlib.pyplot as plt


def relu(signal):
    return np.maximum(0, signal)


def relu_reverse(signal):
    return -np.minimum(0, signal)


x = np.linspace(-10, 10, 2000)
y = np.sin(x)

plt.axhline(color="black")
plt.axvline(color="black")
plt.plot(x, y)
plt.show()

y = relu(y) + relu_reverse(y)

plt.axhline(color="black")
plt.axvline(color="black")
plt.plot(x, y)
plt.show()

x = np.linspace(-10, 10, 2000)

b1 = 0.00 * np.pi
b2 = 0.25 * np.pi
b3 = 0.50 * np.pi
b4 = 0.75 * np.pi
b5 = 1.00 * np.pi
b6 = 1.25 * np.pi
b7 = 1.50 * np.pi
b8 = 1.75 * np.pi

y1 = np.sin(x + b1)
y2 = np.sin(x + b2)
y3 = np.sin(x + b3)
y4 = np.sin(x + b4)
y5 = np.sin(x + b5)
y6 = np.sin(x + b6)
y7 = np.sin(x + b7)
y8 = np.sin(x + b8)

y = relu(y1) + relu(y2) + relu(y3) + relu(y4)
y += relu(y5) + relu(y6) + relu(y7) + relu(y8)

plt.plot(x, y)
plt.axhline(color="black")
plt.axvline(color="black")
plt.show()