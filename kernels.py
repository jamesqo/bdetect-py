import treenode

DOC_INDEX = 0

def _get_tree_kernel_function(name):
    if name == 'ptk':
        return PTKernel()

    raise ValueError(f"Unrecognized tree kernel '{name}'")

class TweetKernel(object):
    def __init__(self, trees, tree_kernel):
        self.trees = trees
        self.tree_kernel = tree_kernel
        self._tree_kernel_function = _get_tree_kernel_function(name=tree_kernel)

    def __call__(self, a, b):
        indexa, indexb = int(a[DOC_INDEX]), int(b[DOC_INDEX])
        treea, treeb = self.trees[indexa], self.trees[indexb]
        return self._tree_kernel_function(treea, treeb)

class PTKernel(object):
    def __init__(self, lambda_, mu):
        self.lambda_ = lambda_
        self.mu = mu
        self._lambda2 = lambda_ ** 2

    def __call__(self, treea, treeb):
        result = 0
        node_pairs = treenode.matching_descendants(treea, treeb)
        for a, b in node_pairs:
            result += self._delta(a, b)
        return result

    def _delta(self, a, b):
        if a.getOutdegree() == 0 or b.getOutdegree() == 0:
            result = self.mu * self._lambda2
        else:
            nca, ncb = a.getOutdegree(), b.getOutdegree()
            result = self.mu * (self._lambda2 + self._sigma_delta_p(a, b, nca, ncb))
        return result

    def _sigma_delta_p(self, a, b, nca, ncb):
        DPS = [[0] * (ncb + 1)] * (nca + 1)
        DP = [[0] * (ncb + 1)] * (nca + 1)
        kmat = [0] * (nca + 1)

        for i in range(1, nca + 1):
            for j in range(1, ncb + 1):
                if a.getChild(i - 1).getLabel() == b.getChild(j - 1).getLabel():
                    DPS[i][j] = self._delta(a.getChild(i - 1), b.getChild(j - 1))
                    kmat[0] += DPS[i][j]
                else:
                    DPS[i][j] = 0

        for s in range(1, min(nca, ncb)):
            for i in range(nca + 1):
                DP[i][s - 1] = 0
            for j in range(ncb + 1):
                DP[s - 1][j] = 0

            for i in range(s, nca + 1):
                for j in range(s, ncb + 1):
                    DP[i][j] = DPS[i][j] + self.lambda_ * DP[i - 1][j] + \
                               self.lambda_ * DP[i][j - 1] - self._lambda2 * DP[i - 1][j - 1]
                    if a.getChild(i - 1).getLabel() == b.getChild(j - 1).getLabel():
                        DPS[i][j] = self._delta(a.getChild(i - 1), b.getChild(j - 1)) * DP[i - 1][j - 1]
                        kmat[s] += DPS[i][j]

        return sum(kmat)
