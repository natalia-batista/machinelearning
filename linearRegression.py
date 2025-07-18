import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

np.random.seed(10)

X = 100 * np.random.rand(100,1)  # 100 examples(m), 1 feature(n)
y = 2 * X[:,0]  + 2*np.random.randn(100)
X = np.c_[X, np.ones(len(X))]  # incluindo o paramentro bias (b) para ser aprendido

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=10)

weights = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

y_hat = X_test @ weights  # nao é mais necessario fazer y_hat = X_test @ weights + b manualmente

mse_test = (1/len(X_test)) * np.linalg.norm(y_hat - y_test) ** 2

print(f"MSE_test: {mse_test}")
plt.scatter(X_train[:,0],y_train,color='green')
plt.scatter(X_test[:,0],y_test,color='blue')
plt.plot(X_test[:,0],y_hat,color='red')
plt.show()