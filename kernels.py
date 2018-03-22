import logging as log
import numpy as np
import sys

import treenode as tn

def _get_tree_kernel_function(name):
    if name == 'ptk':
        return PTKernel()

    raise ValueError("Unrecognized tree kernel '{}'".format(name))

def _get_tweet_index(row):
    TWEET_INDEX_COL_NO = 0
    return int(row[TWEET_INDEX_COL_NO])

class DeltaCache(object):
    def __init__(self):
        self._dict = {}

    def _internal_key(self, key):
        n1, n2 = key
        return n1.data['id'], n2.data['id']

    def __setitem__(self, key, value):
        key = self._internal_key(key)
        self._dict[key] = value

    def clear(self):
        self._dict.clear()

    def get(self, key, default=None):
        key = self._internal_key(key)
        return self._dict.get(key, default)

class TweetKernel(object):
    def __init__(self, trees, tree_kernel):
        self.trees = list(trees)
        self.tree_kernel = tree_kernel
        self._tree_kernel_function = _get_tree_kernel_function(name=tree_kernel)

    def __call__(self, a, b):
        indexa, indexb = _get_tweet_index(a), _get_tweet_index(b)
        treea, treeb = self.trees[indexa], self.trees[indexb]
        return self._tree_kernel_function(treea, treeb)

class PTKernel(object):
    def __init__(self, lambda_=0.4, mu=0.4, normalize=True):
        self.lambda_ = lambda_
        self.mu = mu
        self._lambda2 = lambda_ ** 2
        self._mu_lambda2 = mu * self._lambda2

        self.normalize = normalize
        self._delta_cache = DeltaCache()

    def __call__(self, treea, treeb):
        self._delta_cache.clear()
        k = self._kernel_no_normalize
        if not self.normalize:
            return k(treea, treeb)
        denom = np.sqrt(k(treea, treea)) * np.sqrt(k(treeb, treeb))
        assert denom > 0
        return k(treea, treeb) / denom

    def _kernel_no_normalize(self, treea, treeb):
        result = 0
        node_pairs = tn.matching_descendants(treea, treeb)
        for a, b in node_pairs:
            result += self._delta(a, b)
        return result

    def _delta(self, a, b):
        key = (a, b)
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
