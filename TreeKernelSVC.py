import numpy as np

from sklearn.svm import SVC

from TweetParser import TweetParser
from util import log_mcall

from kernels.PTKernel import PTKernel

class TreeKernelSVC(object):
    def __init__(self, kernel, *args, **kwargs):
        self.kernel = self._get_kernel_function(name=kernel)
        self._parser = TweetParser()
        self._svc = SVC(kernel='precomputed', *args, **kwargs)

    def _get_kernel_function(self, name):
        if name == 'ptk':
            return PTKernel()

        raise ValueError(f"Unrecognized kernel '{name}'")

    def _parse_trees(self, X):
        log_mcall()
        return X['text'].apply(self._parser.tree)

    def _compute_kernel_matrix(self, trees):
        log_mcall()

        m = len(trees)
        m_train = len(self.trees_)
        matrix = np.zeros((m, m_train))

        for i in range(m):
            for j in range(m_train):
                matrix[i, j] = self.kernel(trees[i], self.trees_[j])

        return matrix

    def fit(self, X, y):
        self.X_ = X
        self.trees_ = self._parse_trees(X)
        self.kernel_matrix_ = self._compute_kernel_matrix(self.trees_)
        self._svc.fit(self.kernel_matrix_, y)
        return self

    def predict(self, X):
        trees = self._parse_trees(X)
        kernel_matrix = self._compute_kernel_matrix(trees)
        return self._svc.predict(kernel_matrix)