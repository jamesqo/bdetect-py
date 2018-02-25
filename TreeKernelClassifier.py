import spacy

class TreeKernelClassifier(object):
    def __init__(self, kernel):
        self.kernel = kernel
        self.nlp = spacy.load('en')

    def fit(self, X, y):
        pass
