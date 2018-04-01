import numpy as np
import warnings

from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.svm import SVC

from kernels import TreeKernel
from util import log_call

class TreeSVC(BaseEstimator):
    def __init__(self, estimator, kernel, trees, lambda_=0.4, mu=0.4, normalize=True, ignore_warnings=[]):
        if not isinstance(estimator, SVC):
            raise TypeError(
                "'estimator' should be SVC but instead was {}".format(
                    type(estimator).__name__))
        estimator.kernel = 'precomputed'

        self.estimator = estimator
        self.kernel = kernel
        self.lambda_ = lambda_
        self.mu = mu
        self.normalize = normalize
        self.trees = trees
        self.ignore_warnings = ignore_warnings

        self._X = None
        self.kernel_matrix_ = None
        self._ignored_warnings = False

    @property
    def _kernel_function(self):
        return TreeKernel(name=self.kernel,
                          trees=self.trees,
                          lambda_=self.lambda_,
                          mu=self.mu,
                          normalize=self.normalize)

    def _ignore_warnings(self):
        # NOTE: These filters are not undone on purpose.
        # The reason is, the warnings we want to ignore only appear when grid search is being done in parallel.
        # The code that runs this method in parallel is in Scikit-Learn, outside of our control.
        # Since warnings.catch_warnings() does not work across processes, our only option is to filter warnings inside
        # the code being run in parallel that we control, and not undo the filters so it affects the Scikit-Learn code.
        if not self._ignored_warnings:
            for category in self.ignore_warnings:
                warnings.simplefilter('ignore', category=category)
            self._ignored_warnings = True

    def fit(self, X, y, n_jobs=1, savepath=None):
        self._ignore_warnings()

        self._X = X
        self.kernel_matrix_ = pairwise_kernels(X, self._X, metric=self._kernel_function, n_jobs=n_jobs)
        if savepath is not None:
            np.savetxt(savepath, self.kernel_matrix_, fmt='%g', delimiter=',')
        self.estimator.fit(self.kernel_matrix_, y)
        return self

    def predict(self, X, n_jobs=1, savepath=None):
        kernel_matrix = pairwise_kernels(X, self._X, metric=self._kernel_function, n_jobs=n_jobs)
        if savepath is not None:
            np.savetxt(savepath, kernel_matrix, fmt='%g', delimiter=',')
        return self.estimator.predict(kernel_matrix)
