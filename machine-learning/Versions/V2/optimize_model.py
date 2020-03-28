#!/usr/bin/env python

# This script optimizes the model

import hyperopt
import json
import numpy as np
import pandas as pd
import pickle
import warnings

from functools import partial
from sklearn import datasets, linear_model, svm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

warnings.filterwarnings('ignore')


# Search space for hyperparameter optimization
space = {
    # 'kernel': hyperopt.hp.choice('kernel', ['rbf']),  # 'linear', 'sigmoid', 'poly', 'rbf']
    'C': hyperopt.hp.lognormal('C', 0.0, 2.0),
    'epsilon': hyperopt.hp.uniform('epsilon', 0.01, 0.8),
    'gamma': hyperopt.hp.uniform('gamma', 0, 20)
}


def load_xy(filename):
    df = pd.read_excel(filename, index_col=0)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X.values, y.values


def evaluate(model, X_train, y_train, X_test, y_test):
    #lab_enc = preprocessing.LabelEncoder()
    #encoded = lab_enc.fit_transform(y_train)
    obj = model.fit(X_train, y_train)
    y_pred = obj.predict(X_test)

    return mean_squared_error(y_test, y_pred)


def kfold_score(model, X, y, n_splits=10):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=12345)

    results = []
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        results.append(
            evaluate(model, X_train, y_train, X_test, y_test)
        )
    return np.mean(results)


def fn_to_optimize(params, X, y, **kwargs):
    model = svm.SVR(**params)
    return kfold_score(model, X, y, **kwargs)


def optimize(space, X, y, max_evals=10):
    trials = hyperopt.Trials()

    result = hyperopt.fmin(
        fn=partial(fn_to_optimize, X=X, y=y),
        space=space,
        algo=hyperopt.tpe.suggest,
        trials=trials,
        max_evals=max_evals,
        rstate=np.random.RandomState(12345)
    )
    df_x = pd.DataFrame(trials.idxs_vals[1])
    loss = pd.DataFrame(trials.results)
    df_r = pd.concat((df_x, loss), axis=1)

    return {
        'df': df_r,
        'result': result,
        'trials': trials
    }


def run_once(X, y, C=1.0, epsilon=0.1, **kwargs):
    params = {
        'C': C,
        'epsilon': epsilon
    }
    return fn_to_optimize(params, X, y)


if __name__ == "__main__":
    X, y = load_xy('main_data.xlsx')
    rmse = run_once(X, y, n_folds=10)
    print(rmse)

    r = optimize(space, X, y, max_evals=50)
    df = r['df']
    result = r['result']
    trials = r['trials']
    print(df)

    print("----- Final optimized parameters -----")
    json_str = json.dumps(result, indent=2)
    print(json_str)

    with open('optimized_parameters.json', 'w') as f:
        f.write(json_str)
        print('Saved optimized_parameters.json')
