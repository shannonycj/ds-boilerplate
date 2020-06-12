import os
import json
import hashlib
import shutil
from datetime import datetime

import joblib
import pandas as pd
import numpy as np
import sklearn.preprocessing as sk_preprocessing
import sklearn.metrics as sk_metrics
import matplotlib.pyplot as plt

from config import __config


def regularize_columns(df):
    col_name_regularizer = lambda c: c.lower().replace(' ', '_')
    df.columns = [*map(col_name_regularizer, df.columns.tolist())]
    return df


def detect_id_col(df):
    for col in df.columns:
        if df[col].hasnans:
            continue
        elif pd.api.types.is_numeric_dtype(df[col]):
            if len(set(df[col].tolist())) == len(df) and ('ID' in col or 'Id' in col):
                print(f'Found id col: {col}')
                try:
                    df.set_index(col, inplace=True)
                    print(f'Set {col} as index column')
                    return df
                except Exception as e:
                    print(f'Setting {col} as index failed')
    print('No valid id column found.')
    return df


def split_feature_target(df, target_col):
    fea_cols = set(df.columns) - set([target_col])
    X = df[fea_cols].copy()
    y = df[target_col].copy()
    return X, y


def get_feature_types(df_fea):
    feature_columns = set(df_fea.columns.tolist())
    is_num_col = lambda col: pd.api.types.is_numeric_dtype(df_fea[col])
    num_cols = [*filter(is_num_col, feature_columns)]
    cat_cols = list(feature_columns - set(num_cols))
    return num_cols, cat_cols


def hash_time(digits=10):
    dt = str(datetime.now().timestamp()).encode('utf-8')
    return hashlib.sha256(dt).hexdigest()[:digits]


def init_model_dir(model_name=None):
    model_name = hash_time() if model_name is None else model_name
    path = os.path.join(__config['model_path'], model_name)
    if os.path.isdir(path):
        shutil.rmtree(path)
    try:
        os.mkdir(path)
        return path
    except Exception as e:
        print('Creating model dir failed')
        print(f'Error: {e}')
        return None


def cat_processing(X, cat_cols, max_onehot):
    one_hot, label, cat_features = [], [], []
    encoders = {}
    for fea in cat_cols:
        X[fea] = X[fea].fillna('UNK').astype(str)
        card = X[fea].nunique()
        if card > max_onehot:
            enc = sk_preprocessing.LabelEncoder()
            X[fea] = enc.fit_transform(X[fea])
            label.append(fea)
            cat_features.append(fea)
        else:
            enc = sk_preprocessing.OneHotEncoder(handle_unknown='ignore')
            res = enc.fit_transform(X[fea].values.reshape(-1,1))
            feature_names = enc.get_feature_names([fea])
            for n, v in zip(feature_names, res.toarray().T):
                X[n] = v
            one_hot.append(fea)
            cat_features.extend(feature_names)
        encoders[fea] = enc
    processor_info = {
        'one_hot': one_hot,
        'label': label,
        'columns': cat_features,
        'encoders': encoders
    }
    return X, processor_info


def num_processing(X, num_cols, normalize=False):
    medians = {}
    scaler = None
    for fea in num_cols:
        medians[fea] = X[fea].median()
        X.loc[X[fea].isnull(), fea] = medians[fea]
    if normalize:
        scaler = sk_preprocessing.StandardScaler()
        res = scaler.fit_transform(X[num_cols].values).T
        for i, n in enumerate(num_cols):
            X[n] = res[i]
    processor_info = {
        'columns': num_cols,
        'medians': medians,
        'scaler': scaler
    }
    return X, processor_info


def rf_params():
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    max_features = ['auto', 'sqrt', 'log2']
    max_depth = [3, 5, 8, 10, 15, 20, 30, 50]
    max_depth.append(None)
    min_samples_split = [2, 5, 10, 20]
    min_samples_leaf = [1, 2, 4, 6, 9]
    bootstrap = [True, False]
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap,
                   'criterion': ['gini', 'entropy'],
                   'ccp_alpha': [0., 0.1, 0.5, 1.]}
    return random_grid


def eval_clssifier(clf, X_test, y_true):
    probs = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = sk_metrics.roc_curve(y_true, probs)
    auc = sk_metrics.auc(fpr, tpr)
    print(f'AUC: {auc}')
    plt.plot(fpr, tpr, label='ROC', lw=2)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.show()