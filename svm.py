import numpy as np
import spacy

from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.svm import SVC

from kernels import PTKernel
from util import log_mcall

class TreeKernelSVC(object):
    def __init__(self, kernel, nlp_model='en', *args, **kwargs):
        self.kernel = kernel
        self._kernel_function = self._get_kernel_function(name=kernel)
        self._nlp = spacy.load(nlp_model)
        self._svc = SVC(kernel='precomputed', *args, **kwargs)

    def _get_kernel_function(self, name):
        if name == 'ptk':
            return PTKernel()

        raise ValueError(f"Unrecognized kernel '{name}'")

    def _parse_docs(self, X):
        log_mcall()
        return X['text'].apply(self._nlp)

    def _compute_kernel_matrix(self, docs):
        log_mcall()
        return pairwise_kernels(X=docs,
                                Y=self.docs_,
                                metric=self._kernel_function)

    def fit(self, X, y):
        self.X_ = X
        self.docs_ = self._parse_docs(X)
        self.kernel_matrix_ = self._compute_kernel_matrix(self.docs_)
        self._svc.fit(self.kernel_matrix_, y)
        return self

    def predict(self, X):
        docs = self._parse_docs(X)
        kernel_matrix = self._compute_kernel_matrix(docs)
        return self._svc.predict(kernel_matrix)
