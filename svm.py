import numpy as np

from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.svm import SVC

from kernels import TweetKernel
from util import log_call

class TweetSVC(object):
    def __init__(self, trees, tree_kernel, *args, **kwargs):
        self._kernel = TweetKernel(trees, tree_kernel)
        self._svc = SVC(kernel='precomputed', *args, **kwargs)

    def fit(self, X, y, savepath=None, n_jobs=-1):
        log_call()
        self._X = X
        self.kernel_matrix_ = pairwise_kernels(X, self._X, metric=self._kernel, n_jobs=n_jobs)
        if savepath is not None:
            np.savetxt(savepath, self.kernel_matrix_, fmt='%g', delimiter=',')
        self._svc.fit(self.kernel_matrix_, y)
        return self

    def predict(self, X, savepath=None, n_jobs=-1):
        log_call()
        kernel_matrix = pairwise_kernels(X, self._X, metric=self._kernel, n_jobs=n_jobs)
        if savepath is not None:
            np.savetxt(savepath, kernel_matrix, fmt='%g', delimiter=',')
        return self._svc.predict(kernel_matrix)
