class TweetKernel(object):
    def __init__(self, docs, tree_kernel):
        self.docs = docs
        self.tree_kernel = tree_kernel

    def __call__(self, x, y):
        x = self.docs[x]
        y = self.docs[y]
        return 0

class PTKernel(object):
    def __init__(self):
        pass

    def __call__(self, x, y):
        pass
