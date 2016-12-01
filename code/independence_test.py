# -*- coding: utf-8 -*-
# @Author: Maria Elena Villalobos Ponte
# @Date:   2016-11-22 20:41:39
# @Last Modified by:   Maria Elena Villalobos Ponte
# @Last Modified time: 2016-12-01 14:26:03
import numpy as np
from scipy.stats import gamma
from itertools import permutations, combinations
from sklearn.metrics.pairwise import rbf_kernel, laplacian_kernel


class HSIC_b:
    def __init__(self, X, Y, kernel='exponential'):
        self.n = len(X)
        if kernel == 'exponential':
            apply_kernel = rbf_kernel
        elif kernel == 'laplacian':
            apply_kernel = laplacian_kernel

        # Set kernel variance to median distance between points
        gamma_X = self.__median_dist(X)
        gamma_Y = self.__median_dist(Y)
        # gamma_X, gamma_Y = None, None

        self.K = apply_kernel(X.reshape(-1, 1), gamma=gamma_X)
        self.L = apply_kernel(Y.reshape(-1, 1), gamma=gamma_Y)
        self.H = np.eye(self.n) - np.ones((self.n, self.n)) * (1.0 / self.n)
        self._expected_value = None
        self._variance = None
        self._p_value = None

    @staticmethod
    def __median_dist(X, max_points=100):
        """Median distance between datapoints calculated for at most 100 elements"""
        if len(X) < max_points:
            max_points = len(X)
        res = np.median([np.abs(x1 - x2) for x1, x2 in combinations(X[:100], 2)])
        return res

    def empirical_test(self):
        """Calculated empirical test value"""
        test = np.trace(np.dot(np.dot(np.dot(self.K, self.H), self.L), self.H)) / (self.n ** 2)
        return test

    @staticmethod
    def __mean_squared_norm(kernel):
        n = len(kernel)
        samples_wo_replacement = list(permutations(range(n), 2))
        n_2 = len(samples_wo_replacement)
        sum_pairs = np.array([kernel[ix] for ix in samples_wo_replacement]).sum()
        res = n_2**(-1) * sum_pairs
        return res

    @property
    def expected_value(self):
        if not self._expected_value:
            mu_x_norm = self.__mean_squared_norm(self.K)
            mu_y_norm = self.__mean_squared_norm(self.L)
            res = (1.0 / self.n) * \
                  (1 + mu_x_norm * mu_y_norm - mu_x_norm - mu_y_norm)
            self._expected_value = res
        return self._expected_value

    @property
    def variance(self):
        if not self._variance:
            num = (2.0 * (self.n - 4) * (self.n - 5))
            den = ((self.n) * (self.n - 1) * (self.n - 2) * (self.n - 3))
            scaling = num / den
            HKH = np.dot(np.dot(self.H, self.K), self.H)
            HLH = np.dot(np.dot(self.H, self.L), self.H)
            B = (HKH * HLH) ** 2
            one = np.ones(len(B))
            B_diag = np.diag(np.diag(B))
            B_term = np.dot(np.dot(one, (B - B_diag)), one)
            res = scaling * B_term
            self._variance = res
        return self._variance

    def alpha(self):
        return self.expected_value ** 2 / self.variance

    def beta(self):
        return self.n * self.variance / self.expected_value

    @property
    def p_value(self):
        if not self._p_value:
            a = self.alpha()
            b = self.beta()
            res = gamma.cdf(self.n * self.empirical_test(), a, scale=b)
            self._p_value = res
        return self._p_value

