# %%
import numpy as np

from typing import Any, Dict, List, Optional, Union
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.base import clone, RegressorMixin, BaseEstimator
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    WhiteKernel,
    Matern,
    RationalQuadratic,
    ExpSineSquared,
    DotProduct,
    PairwiseKernel,
)

from julearn.utils import logger
from julearn.model_selection import (
    RepeatedContinuousStratifiedKFold,
    ContinuousStratifiedKFold,
)


class fastGaussianProcessRegressor(RegressorMixin, BaseEstimator):
    """
    Gaussian process regression (GPR) divide and conquer algorithm
    Given K fit K models using one fold at a time
    The final prediction is the average of prediction of all K models
    """

    def __init__(self, n_splits=5, n_repeats=1, stratified=False, **kwargs):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.stratified = stratified
        self.gpr_model = GaussianProcessRegressor(**kwargs)

    def fit(self, X, y):
        X_ = X
        if not isinstance(X, np.ndarray):
            X_ = np.array(X)
        y_ = y
        if not isinstance(y, np.ndarray):
            y_ = np.array(y)
        if self.n_repeats > 1:
            if self.stratified == True:
                logger.info("Using stratified splits (repeated)")
                folds = RepeatedContinuousStratifiedKFold(
                    n_splits=self.n_splits,
                    n_repeats=self.n_repeats,
                    method="quantile",
                    n_bins=10,
                )
            else:
                folds = RepeatedKFold(
                    n_splits=self.n_splits, n_repeats=self.n_repeats)
        else:
            if self.stratified == True:
                logger.info("Using stratified splits")
                folds = ContinuousStratifiedKFold(
                    n_splits=self.n_splits, method="quantile", n_bins=10
                )
            else:
                folds = KFold(n_splits=self.n_splits)
        logger.info(f"Using {folds}")
        # bins = # partition into bins the y
        # folds = StratifiedKFold(n_splits=self.n_splits)
        # Use folds.split(X, bins)
        models = []
        for _, test_index in folds.split(X, y):
            t_model = clone(self.gpr_model)
            t_model.fit(X_[test_index], y_[test_index])
            models.append(t_model)
        logger.info(f"Fitted {len(models)}")
        self.models = models

    def predict(self, X):
        predictions = []
        for model in self.models:
            predictions.append(model.predict(X))
        return np.mean(predictions, axis=0)

    def get_params(self, deep=True):
        out = dict(
            **self.gpr_model.get_params(deep=deep),
            n_splits=self.n_splits,
            n_repeats=self.n_repeats,
            stratified=self.stratified,
        )
        return out

    def set_params(self, **kwargs):
        """Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as :class:`sklearn.pipeline.Pipeline`). The latter have
        parameters of the form ``<component>__<parameter>`` so that it's
        possible to update each component of a nested object.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        gpr_params = list(self.gpr_model.get_params(True).keys())

        for param, val in kwargs.items():
            if param in gpr_params:
                self.gpr_model.set_params(**{param: val})
            else:
                setattr(self, param, val)
        return self
