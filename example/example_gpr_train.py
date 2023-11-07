# iterative GPR algorithm
#%%
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import KFold
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, Matern, RationalQuadratic, ExpSineSquared, DotProduct, PairwiseKernel

class GaussianProcessRegressorIterative(GaussianProcessRegressor):   
    """
    Gaussian process regression (GPR) iterative algorithm
    Given K fit K models iteratively using one fold at a time
    The kernel parameters are updated at each iteration
    """
    def fit(self, X, y, K=5):
        self.K = K
        folds = KFold(n_splits=K)          
        for train_index, test_index in folds.split(X):
            super().fit(X[test_index], y[test_index])
            
        folds = KFold(n_splits=K*2)          
        for train_index, test_index in folds.split(X):
            super().fit(X[test_index], y[test_index])
        

# test it
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.datasets import make_regression, make_friedman2

kernel = ConstantKernel() * DotProduct() + WhiteKernel()


# make data
#X, y = make_regression(n_samples=1000, n_features=2, noise=1, random_state=42)
X, y = make_friedman2(n_samples=1000, noise=0.5)


# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# fit model
model = GaussianProcessRegressorIterative(kernel=kernel)
model.fit(X_train, y_train)

# predict
y_pred = model.predict(X_test)

# evaluate
print(mean_absolute_error(y_test, y_pred))

# plot
plt.scatter(y_test, y_pred)

# %%
# fit model
model_all = GaussianProcessRegressor(kernel=kernel)
model_all.fit(X_train, y_train)

# predict
y_pred_all = model_all.predict(X_test)

# evaluate
print(mean_absolute_error(y_test, y_pred_all))

# plot
plt.scatter(y_test, y_pred_all)
# %%
