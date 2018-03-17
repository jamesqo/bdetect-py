from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.svm import SVC

from kernels import TweetKernel
from util import log_mcall

class TweetSVC(object):
    def __init__(self, trees, tree_kernel, *args, **kwargs):
        self._kernel = TweetKernel(trees, tree_kernel)
        self._svc = SVC(kernel='precomputed', *args, **kwargs)

    def fit(self, X, y):
        log_mcall()
        self._X = X
        self.kernel_matrix_ = pairwise_kernels(X, self._X, metric=self._kernel)
        self._svc.fit(self.kernel_matrix_, y)
        return self

    def predict(self, X):
        log_mcall()
        kernel_matrix = pairwise_kernels(X, self._X, metric=self._kernel)
        return self._svc.predict(kernel_matrix)
