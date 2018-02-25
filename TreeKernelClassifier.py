from nltk.parse.stanford import StanfordDependencyParser

class TreeKernelClassifier(object):
    def __init__(self, jar_path, models_jar_path, kernel):
        self.jar_path = jar_path
        self.models_jar_path = models_jar_path
        self.kernel = kernel

    def fit(self, X, y):
        parser = StanfordDependencyParser(path_to_jar=self.jar_path,
                                          path_to_models_jar=self.models_jar_path)
        tree = parser.raw_parse('Python is such a terrible and verbose programming language. It has 0 Facebook friends.')
        print(type(tree))
        pass
