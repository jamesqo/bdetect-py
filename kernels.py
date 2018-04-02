import logging as log
import numpy as np
import sys

import treenode as tn

def _get_tree_kernel(name, **kwargs):
    if name == 'ptk':
        return PTKernel(**kwargs)

    raise ValueError("Unrecognized tree kernel: {}".format(name))

def _get_tweet_index(row):
    TWEET_INDEX_COL_NO = 0
    return int(row[TWEET_INDEX_COL_NO])

def _get_delta_cache_key(n1, n2):
    return n1.data['id'], n2.data['id']

class TreeKernel(object):
    def __init__(self, name, **kwargs):
        self.name = name
        self._real_kernel = _get_tree_kernel(name, **kwargs)

    def __call__(self, a, b):
        indexa, indexb = _get_tweet_index(a), _get_tweet_index(b)
        return self._real_kernel(indexa, indexb)

class PTKernel(object):
    def __init__(self, trees, lambda_, mu, normalize=True):
        self.lambda_ = lambda_
        self.mu = mu
        self._lambda2 = lambda_ ** 2
        self._mu_lambda2 = mu * self._lambda2

        self.trees = trees
        self.normalize = normalize
        self._delta_cache = {}
        self._idfs = self._compute_idfs(trees)
        if normalize:
            self._sqrt_k_cache = self._compute_sqrt_ks(trees)

    def _compute_idfs(self, trees):
        dfs = {}
        for tree in trees:
            seen = {}
            for node in tn.get_descendants(tree):
                lemma = node.data['lemma']
                if not seen.get(lemma, False):
                    dfs[lemma] = dfs.get(lemma, 0) + 1
                    seen[lemma] = True
        N = len(trees)
        log10_N = np.log10(N)
        return {lemma: log10_N - np.log10(df) for lemma, df in dfs.items()}

    def _compute_sqrt_ks(self, trees):
        k = self._kernel_no_normalize
        return [np.sqrt(k(tree, tree)) for tree in trees]

    def __call__(self, indexa, indexb):
        treea, treeb = self.trees[indexa], self.trees[indexb]
        k = self._kernel_no_normalize
        if not self.normalize:
            return k(treea, treeb)

        # Kernel normalization formula: K'(x, y) = \frac{K(x, y)}{\sqrt{K(x, x) * K(y, y)}}
        denom = self._sqrt_k_cache[indexa] * self._sqrt_k_cache[indexb]
        assert denom > 0
        return k(treea, treeb) / denom

    def _kernel_no_normalize(self, treea, treeb):
        self._delta_cache.clear()

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

        lemma = a.data['lemma']
        assert lemma == b.data['lemma']
        idf = self._idfs[lemma]
        result = self._mu_lambda2 * idf

        nca, ncb = len(a.children), len(b.children)
        if nca != 0 and ncb != 0:
            result += (self.mu * idf * self._sigma_delta_p(a, b, nca, ncb))

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
