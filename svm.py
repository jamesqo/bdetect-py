import numpy as np

from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.svm import SVC

from kernels import TweetKernel
from util import log_call

class TweetSVC(BaseEstimator):
    def __init__(self, **kwargs):
        ker_kwargs = {k[len('ker_'):]: v for k, v in kwargs.items() if k.startswith('ker_')}
        kwargs = {k: v for k, v in kwargs.items() if not k.startswith('ker_')}

        self._kernel = TweetKernel(**ker_kwargs)
        self._svc = SVC(kernel='precomputed', **kwargs)

    def set_trees(self, trees):
        self._kernel.set_trees(trees)

    def fit(self, X, y, n_jobs=-1, savepath=None):
        log_call()
        self._X = X
        self.kernel_matrix_ = pairwise_kernels(X, self._X, metric=self._kernel, n_jobs=n_jobs)
        if savepath is not None:
            np.savetxt(savepath, self.kernel_matrix_, fmt='%g', delimiter=',')
        self._svc.fit(self.kernel_matrix_, y)
        return self

    def predict(self, X, n_jobs=-1, savepath=None):
        log_call()
        kernel_matrix = pairwise_kernels(X, self._X, metric=self._kernel, n_jobs=n_jobs)
        if savepath is not None:
            np.savetxt(savepath, kernel_matrix, fmt='%g', delimiter=',')
        return self._svc.predict(kernel_matrix)
