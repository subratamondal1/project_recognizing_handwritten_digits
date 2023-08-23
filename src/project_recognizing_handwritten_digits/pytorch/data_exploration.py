import matplotlib.pyplot as plt
import numpy as np
from dataset import x_train, y_train, x_valid, y_valid

# Dataset
print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
print(f"x_valid shape: {x_valid.shape}, y_valid shape: {y_valid.shape}")

plt.imshow(x_train[0].reshape((28, 28)), cmap="gray")
plt.show()

print(x_train.shape)