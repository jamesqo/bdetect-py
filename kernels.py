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
        xindex, yindex = int(x[DOC_INDEX]), int(y[DOC_INDEX])
        xdoc, ydoc = self.docs[xindex], self.docs[yindex]
        return self._tree_kernel_function(xdoc, ydoc)

class PTKernel(object):
    def __init__(self):
        pass

    def __call__(self, x, y):
        return 0
