from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import KFold

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
            super().fit(X[test_index ], y[test_index ]) 
