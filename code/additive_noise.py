
import pandas as pd
import pyGPs
from sklearn.model_selection import train_test_split
from independence_test import *
from r_independence import *
import itertools


def compute_gp_regression(X_train, y_train, X_test):
    model = pyGPs.GPR()
    m = pyGPs.mean.Const(y_train.mean())
    k = pyGPs.cov.RBF()
    model.setPrior(mean=m, kernel=k)
    model.optimize(X_train, y_train)
    y_pred, _, _, _, _ = model.predict(X_test)
    return y_pred


def ANM_algorithm_pairwise(X, y, hsic='r'):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    x_train_column = X_train
    x_test_column = X_test

    y_pred = compute_gp_regression(x_train_column, y_train, x_test_column)
    x_pred = compute_gp_regression(y_train, x_train_column, y_test)

    y_residuals = y_test - y_pred.ravel()  # esto no deberia ser absolute value?
    x_residuals = x_test_column - x_pred.ravel()

    if hsic=='py':
        HSIC_x_to_y = HSIC_b(x_test_column, y_residuals)
        HSIC_y_to_x = HSIC_b(y_test, x_residuals)
        if HSIC_x_to_y.empirical_test()< HSIC_y_to_x.empirical_test():
            direction = 1
        else:
            direction = -1
        return direction

    if hsic=='r':
        HSIC_x_to_y = dHSIC(x_test_column, y_residuals)
        HSIC_y_to_x = dHSIC(y_test, x_residuals)
        if HSIC_x_to_y.statistic< HSIC_y_to_x.statistic:
            direction = 1
        else:
            direction = -1
        return direction


def ANM_algorithm(X, y, hsic='py', alpha=0.05):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    leakage_dict = dict()
    variables = X.columns
    for col in variables:
        x_train_column = np.array(X_train[col])
        x_test_column = np.array(X_test[col])

        y_pred = compute_gp_regression(x_train_column, y_train, x_test_column)
        x_pred = compute_gp_regression(y_train, x_train_column, y_test)

        y_residuals = y_test - y_pred.ravel()  # esto no deberia ser absolute value?
        x_residuals = x_test_column - x_pred.ravel()

        if hsic == 'py':
            HSIC_x_to_y = HSIC_b(x_test_column, y_residuals)
            HSIC_y_to_x = HSIC_b(y_test, x_residuals)
            if HSIC_x_to_y.empirical_test() < HSIC_y_to_x.empirical_test():
                leakage_dir = 'No Leakage'
            else:
                leakage_dir = 'Leakage'
            rank = HSIC_x_to_y.empirical_test() - HSIC_y_to_x.empirical_test()

        if hsic == 'r':
            HSIC_x_to_y = dHSIC(x_test_column, y_residuals)
            HSIC_y_to_x = dHSIC(y_test, x_residuals)
            if HSIC_x_to_y.statistic < HSIC_y_to_x.statistic:
                leakage_dir = 'No Leakage'
            else:
                leakage_dir = 'Leakage'
            rank = HSIC_x_to_y.statistic - HSIC_y_to_x.statistic

        dHSIC_xy = dHSIC(x_train_column, y_train)
        p_val = dHSIC_xy.p_value
        if p_val>alpha:
            leakage_dir = 'Independent'
        row = (leakage_dir, rank, p_val)
        leakage_dict[col] = row
    df = pd.DataFrame(leakage_dict).T
    df.columns = ['Direction', 'Rank', 'P_value Independence X and Y']
    return df.sort_values('Rank', ascending=False)

def ANM_algorithm_with_test(X, y, hsic='py', alpha=0.05):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    leakage_dict = dict()
    variables = X.columns
    for col in variables:
        x_train_column = np.array(X_train[col])
        x_test_column = np.array(X_test[col])

        y_pred = compute_gp_regression(x_train_column, y_train, x_test_column)
        x_pred = compute_gp_regression(y_train, x_train_column, y_test)

        y_residuals = y_test - y_pred.ravel()  # esto no deberia ser absolute value?
        x_residuals = x_test_column - x_pred.ravel()

        if hsic == 'py':
            HSIC_x_to_y = HSIC_b(x_test_column, y_residuals)
            HSIC_y_to_x = HSIC_b(y_test, x_residuals)
            if HSIC_x_to_y.empirical_test() < HSIC_y_to_x.empirical_test():
                leakage_dir = 'No Leakage'
            else:
                leakage_dir = 'Leakage'
            rank = HSIC_x_to_y.empirical_test() - HSIC_y_to_x.empirical_test()

        if hsic == 'r':
            HSIC_x_to_y = dHSIC(x_test_column, y_residuals)
            HSIC_y_to_x = dHSIC(y_test, x_residuals)
            if HSIC_x_to_y.statistic < HSIC_y_to_x.statistic:
                leakage_dir = 'No Leakage'
            else:
                leakage_dir = 'Leakage'
            rank = HSIC_x_to_y.statistic - HSIC_y_to_x.statistic

        dHSIC_xy = dHSIC(x_train_column, y_train)
        p_val = dHSIC_xy.p_value
        if p_val>alpha:
            leakage_dir = 'Independent'

        if leakage_dir == 'Leakage':
            list_candidates = list(variables)
            list_candidates.remove(col)
            list_sets_candidates = [c for i in range(len(list_candidates)) for c in itertools.combinations(list_candidates, i + 1)]
            list_sets_candidates = [list(elem) for elem in list_sets_candidates]
            X['target'] = y
            for set in list_sets_candidates:
                ci = CI(col, 'target', set, X)
                if ci.p_value>alpha:
                    leakage_dir = "Column: "+ col + " is independent of target given " + str(set)
                    break

        row = (leakage_dir, rank, p_val)
        leakage_dict[col] = row
    df = pd.DataFrame(leakage_dict).T
    df.columns = ['Direction', 'Rank', 'P_value Independence X and Y']
    return df.sort_values('Rank', ascending=False)



