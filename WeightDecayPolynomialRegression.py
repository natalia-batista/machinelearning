import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

np.random.seed(10)
lbd = 0.6  # lambda (controls the preference for smaller weights) 
#-> small lbd: (potentially) overfitting -> large lbd: (potentially) underfitting
degree = 9
examples = 100


# GENERATING SYNTHETIC DATA
X = np.random.rand(examples,1)
y = X[:,0]**3 + X[:,0]**2 + 2 * X[:,0] + 0.1 * np.random.randn(examples)

#BUILDING A DESIGN MATRIX
# a way of describing the dataset, one example per row and one feature per column [1, x_i, x_i^2,...x_i^k]
# in linear regression, the design matrix is simply the features column
X_design = np.zeros((len(X), degree + 1))
for i in range(degree + 1):
    X_design[:, i] = X[:,0] ** i  # column i is x^i

# TRAIN/TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(X_design, y, test_size=0.2, shuffle=True, random_state=10)

#COMPUTING WEIGHTS USING UNDERDETERMINED NORMAL EQUATIONS
w = np.linalg.pinv(X_train.T @ X_train + lbd * np.eye(X_train.shape[1])) @ X_train.T @ y_train  
# using the Moore-Penrose pseudo inverse to handle potential ill-conditioning due to high-degree polynomials

#COMPUTING PREDICTION ON TRAINING SET
y_hat = X_train @ w
mse_train = (1/len(X_train)) * np.linalg.norm(y_hat - y_train) ** 2
j = mse_train + lbd * w.T@w

#COMPUTING PREDICTION ON TEST SET
y_test_hat = X_test @ w
mse_test = np.mean((y_test - y_test_hat) ** 2)
j_test = mse_test + lbd * w.T@w

print(f"Training MSE: {mse_train}")
print(f"Test MSE:     {mse_test}")
print(f"Training criterion J(w): {j}")

#model's prediction for every point in this interval to draw the curve
x_curve = np.linspace(0, 1, 200)
X_curve_design = np.zeros((len(x_curve), degree + 1))
for i in range(degree + 1):
    X_curve_design[:, i] = x_curve ** i
y_curve = X_curve_design @ w

plt.scatter(X_train[:,1], y_train, color='green', label='Train data')
plt.scatter(X_test[:,1], y_test, color='blue', label='Test data')
plt.plot(x_curve, y_curve, color='red', label='Fitted curve')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()