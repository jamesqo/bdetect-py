import spacy

from sklearn.svm import SVC

class TreeKernelSVC(object):
    def __init__(self, kernel, *args, **kwargs):
        self.kernel = kernel
        self.nlp = spacy.load('en')

        self._svc = SVC(kernel='precomputed', *args, **kwargs)

    def _compute_kernel_matrix(X):
        pass

    def fit(self, X, y):
        self._compute_kernel_matrix(X)
        self._svc.fit(self.kernel_matrix_, y)
        return self

    def predict(self, X):
        pass
