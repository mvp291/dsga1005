# -*- coding: utf-8 -*-
# @Author: Maria Elena Villalobos Ponte
# @Date:   2016-11-22 20:41:39
# @Last Modified by:   Maria Elena Villalobos Ponte
# @Last Modified time: 2016-12-14 20:07:15
from __future__ import division
import numpy as np
from numpy import linalg as LA
from scipy.stats import gamma
from itertools import permutations, combinations
from sklearn.metrics.pairwise import rbf_kernel, laplacian_kernel

def median_dist(X, max_points=100):
        """ Median distance between datapoints calculated for at most 100 elements
           to use in rbf_kernel method as specified by the median heuristic 
           (Scholkopf and Smola, 2002)
        """
        if len(X) < max_points:
            max_points = len(X)
        res = np.sqrt(0.5 * np.median([np.linalg.norm(x1 - x2) for x1, x2 in combinations(X[:100], 2)]))
        return res

class HSIC_b:
    def __init__(self, X, Y, kernel='exponential'):
        self.n = len(X)
        if kernel == 'exponential':
            apply_kernel = rbf_kernel
        elif kernel == 'laplacian':
            apply_kernel = laplacian_kernel

        # Set kernel variance to median distance between points
        gamma_X = median_dist(X)
        gamma_Y = median_dist(Y)

        self.K = apply_kernel(X.reshape(-1, 1), gamma=gamma_X)
        self.L = apply_kernel(Y.reshape(-1, 1), gamma=gamma_Y)
        H = np.identity(self.n) - np.ones((self.n, self.n), dtype = float) / self.n

        self.HKH = np.dot(np.dot(H, self.K), H)
        self.HLH = np.dot(np.dot(H, self.L), H)

        self._expected_value = None
        self._variance = None
        self._p_value = None

    def empirical_test(self):
        """Calculated empirical test value (HSIC)"""
        HKHLH = np.dot(self.HKH, self.HLH)
        test = np.trace(HKHLH) / ((self.n - 1) ** 2)
        return test

    @staticmethod
    def __mean_squared_norm(kernel):
        n = kernel.shape[0]
        n_2 = n * (n - 1)
        res = (np.sum(kernel) - np.trace(kernel)) / n_2
        return res

    @property
    def expected_value(self):
        if not self._expected_value:
            mu_x_norm = self.__mean_squared_norm(self.K)
            mu_y_norm = self.__mean_squared_norm(self.L)
            res = (1 + mu_x_norm * mu_y_norm - mu_x_norm - mu_y_norm) / self.n
            self._expected_value = res
        return self._expected_value

    @property
    def variance(self):
        if not self._variance:
            num = 2.0 * (self.n - 4) * (self.n - 5)
            den = (self.n) * (self.n - 1) * (self.n - 2) * (self.n - 3) * ((self.n-1) ** 4)
            scaling = num / den
            B = (self.HKH * self.HLH) ** 2
            # B_diag = np.diag(np.diag(B))
            B_term = (np.sum(B) - np.trace(B))
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
            res = gamma.sf(self.n * self.empirical_test(), a, scale=b)
            self._p_value = res
        return self._p_value
