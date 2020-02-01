from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from sklearn.metrics import adjusted_rand_score
from scipy.io import savemat, loadmat
from subprocess import Popen, PIPE
from tcorex.covariance import calculate_nll_score
from .data import make_buckets

import numpy as np
import time
import itertools
import random
import re
import os


class Baseline(object):
    def __init__(self, name):
        self.name = name
        self._trained = False
        self._val_score = None
        self._params = None
        self._clusters = None
        self._method = None

    def select(self, train_data, true_clusters, params, verbose=True):
        if verbose:
            print("\n{}\nSelecting the best parameter values for {} ...".format('-' * 80, self.name))

        best_score = -1e18
        best_params = None
        best_clusters= None
        results = []

        const_params, grid = self.make_param_grid(params)

        for index, cur_params in enumerate(grid):
            if verbose:
                self.print_progress(cur_params, grid, index)

            cur_params = dict(cur_params)
            for k, v in const_params.items():
                cur_params[k] = v

            # divide into buckets if needed
            try:
                cur_clusters = self._train(train_data, cur_params, verbose)
                cur_score = adjusted_rand_score(true_clusters, cur_clusters)
            except Exception as e:
                print("Failed to train and evaluate method: {}, message: {}".format(self.name, str(e)))
                cur_score = None
                cur_clusters = None
            results.append((cur_params, cur_score))

            if verbose:
                print('\tcurrent score: {}'.format(cur_score))

            if (best_params is None) or (not np.isnan(cur_score) and cur_score < best_score):
                best_score = cur_score
                best_params = cur_params
                best_clusters = cur_clusters
        if verbose:
            print('\nFinished with best validation score: {}'.format(best_score))

        self._trained = True
        self._val_score = best_score
        self._params = best_params
        self._clusters = best_clusters

        return best_score, best_params, best_clusters, results

    def print_progress(self, cur_params, grid, index):
        print("done {} / {}".format(index, len(grid)), end='')
        print(" | running with ", end='')
        for k, v in cur_params:
            if k != '__dummy__':
                print('{}: {}\t'.format(k, v), end='')
        print('')

    def make_param_grid(self, params):
        const_params = dict()
        search_params = []
        for k, v in params.items():
            if isinstance(v, list):
                arr = [(k, x) for x in v]
                search_params.append(arr)
            elif isinstance(v, dict):
                arr = []
                for param_k, param_v in v.items():
                    arr += list([(param_k, x) for x in param_v])
                search_params.append(arr)
            else:
                const_params[k] = v
        # add a dummy variable if the grid is empty
        if len(search_params) == 0:
            search_params = [[('__dummy__', None)]]
        grid = list(itertools.product(*search_params))
        return const_params, grid

    def _train(self, train_data, params, verbose) -> np.ndarray:
        raise NotImplementedError()

    def get_covariance(self):
        assert self._trained
        return self._covs


class PCA(Baseline):
    def __init__(self, **kwargs):
        super(PCA, self).__init__(**kwargs)

    def _train(self, data, params, verbose):
        import sklearn.decomposition as sk_dec
        if verbose:
            print("Training {} ...".format(self.name))
        start_time = time.time()
        try:
            est = sk_dec.PCA(n_components=params['n_components'])
            est.fit(data)
            clusters = np.argmax(est.components_, axis=0)
        except Exception as e:
            clusters = None
            if verbose:
                print(f"\t{self.name} failed with message: {e}")
        finish_time = time.time()
        if verbose:
            print("\tElapsed time {:.1f}s".format(finish_time - start_time))
        return clusters


class ICA(Baseline):
    def __init__(self, **kwargs):
        super(ICA, self).__init__(**kwargs)

    def _train(self, data, params, verbose):
        import sklearn.decomposition as sk_dec
        if verbose:
            print("Training {} ...".format(self.name))
        start_time = time.time()
        try:
            est = sk_dec.FastICA(n_components=params['n_components'])
            est.fit(data)
            clusters = np.argmax(est.components_, axis=0)
        except Exception as e:
            clusters = None
            if verbose:
                print(f"\t{self.name} failed with message: {e}")
        finish_time = time.time()
        if verbose:
            print("\tElapsed time {:.1f}s".format(finish_time - start_time))
        return clusters


class FactorAnalysis(Baseline):
    def __init__(self, **kwargs):
        super(FactorAnalysis, self).__init__(**kwargs)

    def _train(self, data, params, verbose):

        def _rotate(components, n_components, method="varimax", tol=1e-6):
            """Rotate the factor analysis solution."""
            implemented = ("varimax", "quartimax")
            if method in implemented:
                return _ortho_rotation(components.T, method=method, tol=tol)[:n_components]
            else:
                raise ValueError("'method' must be in %s, not %s"
                                 % (implemented, method))

        def _ortho_rotation(components, method='varimax', tol=1e-6, max_iter=100):
            """Return rotated components."""
            nrow, ncol = components.shape
            rotation_matrix = np.eye(ncol)
            var = 0

            for _ in range(max_iter):
                comp_rot = np.dot(components, rotation_matrix)
                if method == "varimax":
                    tmp = np.diag((comp_rot ** 2).sum(axis=0)) / nrow
                    tmp = np.dot(comp_rot, tmp)
                elif method == "quartimax":
                    tmp = 0
                else:
                    raise AttributeError()
                u, s, v = np.linalg.svd(
                    np.dot(components.T, comp_rot ** 3 - tmp))
                rotation_matrix = np.dot(u, v)
                var_new = np.sum(s)
                if var != 0 and var_new < var * (1 + tol):
                    break
                var = var_new

            return np.dot(components, rotation_matrix).T

        import sklearn.decomposition as sk_dec
        if verbose:
            print("Training {} ...".format(self.name))
        start_time = time.time()
        try:
            est = sk_dec.FactorAnalysis(n_components=params['n_components'])
            est.fit(data)
            clusters = np.argmax(_rotate(est.components_, est.n_components), axis=0)
        except Exception as e:
            clusters = None
            if verbose:
                print(f"\t{self.name} failed with message: {e}")
        finish_time = time.time()
        if verbose:
            print("\tElapsed time {:.1f}s".format(finish_time - start_time))
        return clusters


class LinearCorex(Baseline):
    def __init__(self, **kwargs):
        super(LinearCorex, self).__init__(**kwargs)

    def _train(self, data, params, verbose):
        import linearcorex
        if verbose:
            print("Training {} ...".format(self.name))
        start_time = time.time()
        c = linearcorex.Corex(n_hidden=params['n_hidden'],
                              max_iter=params['max_iter'],
                              anneal=params['anneal'])
        c.fit(data)
        clusters = c.mis.argmax(axis=0)
        finish_time = time.time()
        if verbose:
            print("\tElapsed time {:.1f}s".format(finish_time - start_time))
        return clusters


class KMeans(Baseline):
    def __init__(self, **kwargs):
        super(KMeans, self).__init__(**kwargs)

    def _train(self, data, params, verbose):
        from sklearn.cluster import KMeans
        if verbose:
            print("Training {} ...".format(self.name))
        start_time = time.time()
        data = data.T  # We want to cluster variables, not data points
        c = KMeans(n_clusters=params['n_clusters'], random_state=13)
        c.fit(data)
        clusters = c.predict(data)
        finish_time = time.time()
        if verbose:
            print("\tElapsed time {:.1f}s".format(finish_time - start_time))
        return clusters


class Spectral(Baseline):
    def __init__(self, **kwargs):
        super(Spectral, self).__init__(**kwargs)

    def _train(self, data, params, verbose):
        from sklearn.cluster import SpectralClustering
        if verbose:
            print("Training {} ...".format(self.name))
        start_time = time.time()
        data = data.T  # We want to cluster variables, not data points
        c = SpectralClustering(n_clusters=params['n_clusters'], random_state=13)
        c.fit(data)
        clusters = c.labels_
        finish_time = time.time()
        if verbose:
            print("\tElapsed time {:.1f}s".format(finish_time - start_time))
        return clusters


class Hierarchical(Baseline):
    def __init__(self, **kwargs):
        super(Hierarchical, self).__init__(**kwargs)

    def _train(self, data, params, verbose):
        from sklearn.cluster import AgglomerativeClustering
        if verbose:
            print("Training {} ...".format(self.name))
        start_time = time.time()
        data = data.T  # We want to cluster variables, not data points
        c = AgglomerativeClustering(n_clusters=params['n_clusters'])
        c.fit(data)
        clusters = c.labels_
        finish_time = time.time()
        if verbose:
            print("\tElapsed time {:.1f}s".format(finish_time - start_time))
        return clusters
