import matplotlib.pyplot as plt
import numpy as np

# Create numpy arrays
x = np.array([1, 2, 3, 4, 5], dtype=np.float32)
y = np.array([-2, 1, 2, -10, 22], dtype=np.float32)
print("x-Vektor", x)
print("y-Vektor", y)

print(np.max(x, axis=0), np.argmax(x))
print(np.min(x))
print(np.mean(x))
print(np.median(x))

# Scatter and Plot
plt.scatter(x, y, color="red")
plt.plot(x, y, color="blue") # plot
plt.legend(["f(x)"])
plt.title("This is a title")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.show()
