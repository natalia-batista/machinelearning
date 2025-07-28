import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
np.random.seed(10)

# KNN is a non-parametric model, memory-based method

k = 10  # hyperparameter that defines the number of neighbors used 
examples = 100
X = 100 * np.random.rand(examples,1)  # 100 examples(m), 1 feature(n)
y = 2 * X[:,0]  + 2*np.random.randn(examples)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=10)

# calculating the distances and the average of the k nearest neighbors
distances = np.empty((len(X_test),len(X_train)))
y_hat = np.empty(len(y_test))
for i in range(len(X_test)):
    distances[i,:] = np.abs(X_test[i, 0] - X_train[:, 0])
    min_distances = np.argsort(distances[i,:],kind='mergesort')[:k]
    y_hat[i] = np.average(y_train[min_distances])

mse_test = (1/len(X_test)) * np.linalg.norm(y_hat - y_test) ** 2

print(f"MSE_test: {mse_test}")
plt.scatter(X_train[:,0], y_train, color='green', label='Train data')
plt.scatter(X_test[:,0], y_test, color='blue', label='Test data')
plt.scatter(X_test[:,0], y_hat, color='red', label='KNN result')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()