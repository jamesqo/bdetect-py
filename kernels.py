import logging as log
import numpy as np
import sys

import treenode as tn

def _get_tree_kernel(name, **kwargs):
    if name == 'ptk':
        return PTKernel(**kwargs)

    raise ValueError("Unrecognized tree kernel '{}'".format(name))

def _get_tweet_index(row):
    TWEET_INDEX_COL_NO = 0
    return int(row[TWEET_INDEX_COL_NO])

def _get_delta_cache_key(n1, n2):
    return n1.data['id'], n2.data['id']

class TweetKernel(object):
    def __init__(self, name, **kwargs):
        self.name = name
        self._tree_kernel = _get_tree_kernel(name, **kwargs)

    def __call__(self, a, b):
        indexa, indexb = _get_tweet_index(a), _get_tweet_index(b)
        return self._tree_kernel(indexa, indexb)

    def set_trees(self, trees):
        self._tree_kernel.set_trees(trees)

class PTKernel(object):
    def __init__(self, lambda_, mu, normalize=True):
        self.lambda_ = lambda_
        self.mu = mu
        self._lambda2 = lambda_ ** 2
        self._mu_lambda2 = mu * self._lambda2

        self.trees = None
        self.normalize = normalize
        self._delta_cache = {}
        self._sqrt_k_cache = None

    def set_trees(self, trees):
        self.trees = list(trees)
        if self.normalize:
            self._sqrt_k_cache = self._compute_sqrt_ks(trees)

    def _compute_sqrt_ks(self, trees):
        result = []
        # IMPORTANT NOTE: You must clear the delta cache in between multiple calls to k.
        k = self._kernel_no_normalize
        for tree in trees:
            self._delta_cache.clear()
            sqrt_k = np.sqrt(k(tree, tree))
            result.append(sqrt_k)
        return result

    def __call__(self, indexa, indexb):
        self._delta_cache.clear()

        treea, treeb = self.trees[indexa], self.trees[indexb]
        # IMPORTANT NOTE: You must clear the delta cache in between multiple calls to k.
        k = self._kernel_no_normalize
        if not self.normalize:
            return k(treea, treeb)

        # Kernel normalization formula: K'(x, y) = \frac{K(x, y)}{\sqrt{K(x, x) * K(y, y)}}
        denom = self._sqrt_k_cache[indexa] * self._sqrt_k_cache[indexb]
        assert denom > 0
        return k(treea, treeb) / denom

    def _kernel_no_normalize(self, treea, treeb):
        result = 0
        node_pairs = tn.matching_descendants(treea, treeb)
        for a, b in node_pairs:
            result += self._delta(a, b)
        return result

    def _delta(self, a, b):
        key = _get_delta_cache_key(a, b)
        result = self._delta_cache.get(key)
        if result is not None:
            return result

        nca, ncb = len(a.children), len(b.children)
        result = self._mu_lambda2
        if nca != 0 and ncb != 0:
            result += (self.mu * self._sigma_delta_p(a, b, nca, ncb))

        self._delta_cache[key] = result
        return result

    def _sigma_delta_p(self, a, b, nca, ncb):
        DPS = [[0 for i in range(ncb + 1)] for j in range(nca + 1)]
        DP = [[0 for i in range(ncb + 1)] for j in range(nca + 1)]
        kmat = [0] * (nca + 1)

        for i in range(1, nca + 1):
            for j in range(1, ncb + 1):
                if tn.eq(a.children[i - 1], b.children[j - 1]):
                    DPS[i][j] = self._delta(a.children[i - 1], b.children[j - 1])
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
                    if tn.eq(a.children[i - 1], b.children[j - 1]):
                        DPS[i][j] = self._delta(a.children[i - 1], b.children[j - 1]) * DP[i - 1][j - 1]
                        kmat[s] += DPS[i][j]

        return sum(kmat)
