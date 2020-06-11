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
        RandomForestClassifier(), random_grid, n_iter=n_iter, cv=cv, verbose=2,
        random_state=seed, n_jobs=-1)
    grid.fit(X_train, y_train)
    clf = grid.best_estimator_
    print(f'training score: {clf.score(X_train, y_train)}')
    print(f'testing score: {clf.score(X_test, y_test)}')
    print(grid.best_params_)
    utils.eval_clssifier(clf, X_test, y_test)
    return clf, grid.best_params_, RandomForestClassifier


def train_lr_classifier(X, y, n_iter=20, cv=3, seed=1):
    from sklearn.linear_model import LogisticRegression
    params = {
        'penalty': ['l2'],
        'fit_intercept': [True, False],
        'max_iter': [1000, 2000],
        'C': [i * 0.1 for i in range(10)] + [1., 2., 5., 10.],
        'class_weight': [None, 'balanced'],
        'solver': ['lbfgs']
        }
    X_train, X_test, y_train, y_test = sk_model_selection.train_test_split(
        X, y, test_size=.3, random_state=seed)
    grid = sk_model_selection.RandomizedSearchCV(
        LogisticRegression(), params, n_iter=n_iter, cv=cv, verbose=2,
        random_state=seed, n_jobs=-1)
    grid.fit(X_train, y_train)
    clf = grid.best_estimator_
    print(f'training score: {clf.score(X_train, y_train)}')
    print(f'testing score: {clf.score(X_test, y_test)}')
    print(grid.best_params_)
    utils.eval_clssifier(clf, X_test, y_test)
    return clf, grid.best_params_, LogisticRegression


def train_catboost_classifier(df_fea, labels, cat_cols, params=None, plot=True):
    import catboost
    X_train, X_test, y_train, y_test = sk_model_selection.train_test_split(
        df_fea, labels, test_size=.3, random_state=1)
    cat_fea = np.where(X_train.columns.isin(cat_cols))[0]
    if params is None:
        params = {'iterations':5000,
                'learning_rate':0.01,
                'cat_features': cat_fea,
                'depth':3,
                'eval_metric':'AUC',
                'verbose':200,
                'od_type':"Iter", # overfit detector
                'od_wait':500, # most recent best iteration to wait before stopping
                'random_seed': 1
                }
    else:
        params['cat_features'] = cat_fea

    cat_model = catboost.CatBoostClassifier(**params)
    cat_model.fit(X_train, y_train,   
            eval_set=(X_test, y_test), 
            use_best_model=True, # True if we don't want to save trees created after iteration with the best validation score
            plot=plot  
            )
    return cat_model, X_train.columns.tolist()