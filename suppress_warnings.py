import warnings

from sklearn.base import BaseEstimator

class SuppressWarningsEstimator(BaseEstimator):
    def __init__(self, inner, categories):
        self.inner = inner
        self.categories = categories

    def add_filters(self):
        for category in self.categories:
            warnings.simplefilter('ignore', category=category)

    def get_params(self, *args, **kwargs):
        with warnings.catch_warnings():
            self.add_filters()
            return self.inner.get_params(*args, **kwargs)

    def set_params(self, *args, **kwargs):
        with warnings.catch_warnings():
            self.add_filters()
            return self.inner.set_params(*args, **kwargs)

    def fit(self, *args, **kwargs):
        with warnings.catch_warnings():
            self.add_filters()
            return self.inner.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        with warnings.catch_warnings():
            self.add_filters()
            return self.inner.predict(*args, **kwargs)
