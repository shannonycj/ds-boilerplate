import sklearn.model_selection as sk_model_selection
import sklearn.metrics as sk_metrics
import numpy as np

from kernel import utils


def train_rf_regression(X, y, n_iter=100, cv=3):
    from sklearn.ensemble import RandomForestRegressor

    X_train, X_test, y_train, y_test = sk_model_selection.train_test_split(
        X, y, test_size=.3, random_state=1)
    
    params = utils.rf_params()
    grid = sk_model_selection.RandomizedSearchCV(
        RandomForestRegressor(), random_grid, n_iter=n_iter, cv=cv, verbose=2,
        random_state=1, n_jobs=-1)
    grid.fit(X_train, y_train)
    clf = grid.best_estimator_
    print(f'training score: {clf.score(X_train, y_train)}')
    print(f'testing score: {clf.score(X_test, y_test)}')
    return clf, grid.best_params_


def train_rf_classifier(X, y, n_iter=100, cv=3, seed=1):
    from sklearn.ensemble import RandomForestClassifier
    random_grid = utils.rf_params()
    X_train, X_test, y_train, y_test = sk_model_selection.train_test_split(
        X, y, test_size=.3, random_state=seed)
    grid = sk_model_selection.RandomizedSearchCV(
        RandomForestClassifier(), random_grid, n_iter=100, cv=3, verbose=2,
        random_state=1, n_jobs = -1)
    grid.fit(X_train, y_train)
    clf = grid.best_estimator_
    print(f'training score: {clf.score(X_train, y_train)}')
    print(f'testing score: {clf.score(X_test, y_test)}')
    print(grid.best_params_)
    return clf, grid.best_params_, RandomForestClassifier


def train_lr_classifier(X, y, n_iter=20, cv=3, seed=1):
    from sklearn.linear_model import LogisticRegression
    params = {
        'penalty': ['l2'],
        'fit_intercept': [True, False],
        'max_iter': [1000, 2000],
        'C': [i * 0.1 for i in range(10)] + [1., 2., 5., 10.],
        'class_weight': [None, 'balanced']
        }
    X_train, X_test, y_train, y_test = sk_model_selection.train_test_split(
        X, y, test_size=.3, random_state=seed)
    grid = sk_model_selection.RandomizedSearchCV(
        LogisticRegression(solver='lbfgs'), params, n_iter=20, cv=3, verbose=2,
        random_state=1, n_jobs=-1)
    grid.fit(X_train, y_train)
    clf = grid.best_estimator_
    print(f'training score: {clf.score(X_train, y_train)}')
    print(f'testing score: {clf.score(X_test, y_test)}')
    print(grid.best_params_)
    return clf, grid.best_params_, LogisticRegression