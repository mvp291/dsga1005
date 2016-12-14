import numpy as np
import pyGPs
import sklearn
import hsic as hsic2
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import rbf_kernel, laplacian_kernel
from independence_test import *



def compute_gp_regression(X_train, y_train, X_test):
    model = pyGPs.GPR()
    m = pyGPs.mean.Const()
    k = pyGPs.cov.RBF()
    model.setPrior(mean=m, kernel=k)
    model.optimize(X_train, y_train)
    print('Optimized negative log marginal likelihood:', round(model.nlZ,3))
    y_pred, _, _, _, _ = model.predict(X_test)
    return y_pred


def ANM_algorithm(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    leakage_prob = dict()

    for col in range(X_train.shape[1]):
        x_train_column = X_train[:, col]
        print x_train_column
        x_test_column = X_test[:, col]

        y_pred = compute_gp_regression(x_train_column, y_train, x_test_column)
        x_pred = compute_gp_regression(y_train, x_train_column, y_test)

        y_residuals = y_test - y_pred.ravel()  # esto no deberia ser absolute value?
        x_residuals = x_test_column - x_pred.ravel()

        HSIC_x_to_y = HSIC_b(x_test_column, y_residuals)
        HSIC_y_to_x = HSIC_b(y_test, x_residuals)

        HSIC_x_to_y2 , _= hsic2(x_test_column, y_residuals, 0.05)
        HSIC_y_to_x2 , _= hsic2(y_test, x_residuals, 0.05)

        print "Matile" + str(HSIC_x_to_y) + str(HSIC_y_to_x)
        print "Otro" + str(HSIC_x_to_y2) + str(HSIC_y_to_x2)

        if HSIC_x_to_y.empirical_test()< HSIC_y_to_x.empirical_test():
            leakage_prob[col] = 'No Leakage'
        else:
            leakage_prob[col] = 'Leakage'

    for col in leakage_prob:
        print "Column "+ str(col)+ ' was '+ leakage_prob[col]
    """
    keys = leakage_prob.keys()
    keys.sort(reverse=True)
    for key in keys:
        print "The probability of column: " + str(leakage_prob[key]) + " is: " + str(key)
    """