def _get_tree_kernel_function(name):
    if name == 'ptk':
        return PTKernel()

    raise ValueError(f"Unrecognized tree kernel '{name}'")

DOC_INDEX = 0

class TweetKernel(object):
    def __init__(self, docs, tree_kernel):
        self.docs = docs
        self.tree_kernel = tree_kernel
        self._tree_kernel_function = _get_tree_kernel_function(name=tree_kernel)

    def __call__(self, x, y):
        x_doc = self.docs[x[DOC_INDEX]]
        y_doc = self.docs[y[DOC_INDEX]]
        return self._tree_kernel_function(x_doc, y_doc)

class PTKernel(object):
    def __init__(self):
        pass

    def __call__(self, x, y):
        pass
