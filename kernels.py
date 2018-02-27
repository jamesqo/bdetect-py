class TweetKernel(object):
    def __init__(self, tree_kernel):
        self.tree_kernel = tree_kernel

    def __call__(self, x, y):
        print(type(x), type(y))

class PTKernel(object):
    def __init__(self):
        pass

    def __call__(self, x, y):
        pass
