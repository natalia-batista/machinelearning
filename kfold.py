import numpy as np
np.random.seed(10)
# obs: I'm using my linear regression code as a starting point

# REQUIRE: D -> given dataset
# REQUIRE: A -> the learning algorithm
# REQUIRE: L -> loss function
# REQUIRE: k -> number of folds

k = 10 
D = 100 * np.random.rand(100,1)  # 100 examples(m), 1 feature(n)
Dy = 2 * D[:,0]  + 2*np.random.randn(100)
indices = np.random.permutation(len(D))
D = D[indices]
Dy = Dy[indices]
D = np.c_[D, np.ones(len(D))]  # incluindo o paramentro bias (b) para ser aprendido

# split D into k mutually exclusive subsets D_i, whose union is D
D_i = np.array_split(D,k)
Dy_i = np.array_split(Dy,k)

ej = 0

for i in range(k):
    # training set D\Di, f_i = A(D\Di)
    D_train = np.vstack([D_i[j] for j in range(k) if j != i])
    Dy_train = np.hstack([Dy_i[j] for j in range(k) if j != i])
    weights = np.linalg.inv(D_train.T @ D_train) @ D_train.T @ Dy_train
    # calculating the error on the test set Di, L = mse
    y_pred = D_i[i] @ weights
    ej += (1/len(D_i[i])) * np.linalg.norm(y_pred - Dy_i[i]) ** 2
    

print(f'error: {ej/k}')

  



