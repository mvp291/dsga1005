# -*- coding: utf-8 -*-
# @Author: Maria Elena Villalobos Ponte
# @Date:   2016-12-14 22:10:04
# @Last Modified by:   Maria Elena Villalobos Ponte
# @Last Modified time: 2016-12-14 22:24:32
import rpy2
import rpy2.robjects.numpy2ri
import numpy as np
from rpy2.robjects.packages import importr

rpy2.robjects.numpy2ri.activate()

dHSIC_R = importr('dHSIC')

class dHSIC:
    def __init__(self, X, Y, alpha=0.05, method="gamma",
                 kernel="gaussian", B=100, pairwise=False):
        self.res = dHSIC_R.dhsic_test(X, Y, alpha, method, kernel, B, pairwise)
        self.statistic = tuple(self.res[0])[0]
        self.critic_value = tuple(self.res[1])[0]
        self.p_value = tuple(self.res[2])[0]


if __name__ == '__main__':
    pass