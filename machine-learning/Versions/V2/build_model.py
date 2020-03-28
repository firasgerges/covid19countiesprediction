#!/usr/bin/env python

import pandas as pd
import pickle
import warnings

from sklearn import datasets, linear_model, svm
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import preprocessing

warnings.filterwarnings('ignore')

#-------------------------Not Using---------------------------------------
models = {
    'dtr': DecisionTreeRegressor(),
    'linear': linear_model.LinearRegression(),
    'logistic': linear_model.LogisticRegression(),
    'knr': KNeighborsRegressor(),
    'svr': svm.SVR(),
    'svr_rbf': svm.SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1),
    'svr_lin': svm.SVR(kernel='linear', C=100, gamma='auto'),
    'svr_poly': svm.SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1, coef0=1),
    'gbr': GradientBoostingRegressor(),
    'rf': RandomForestRegressor(n_estimators=100, criterion='mse')
}

#-------------------------------------------------------------------------

models = {
    'svr': svm.SVR(gamma='scale', C=1.0, epsilon=0.2),
    'svr_opt1': svm.SVR(gamma='scale', C=0.17654433034967923, epsilon=0.06953152985642766),  # Optimized models
    'svr_opt2': svm.SVR(gamma=15.157706042831762, C=90.73680833133838, epsilon=0.7047413056425894)
}

scores = {
    'MSE': mean_squared_error,
    'MAE': mean_absolute_error,
    'r2': r2_score
}

def load_xy(filename):
    df = pd.read_excel(filename, index_col=0)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X.values, y.values


def evaluate(model, X_train, y_train, X_test, y_test, scores, use_scaler=True):
    #lab_enc = preprocessing.LabelEncoder()
    #encoded = lab_enc.fit_transform(y_train)

    if use_scaler is True:
        scaler = preprocessing.StandardScaler(with_mean=False)
        y_train_s = scaler.fit_transform(y_train.reshape(-1,1)).flatten()
        y_test_s = scaler.transform(y_test.reshape(-1,1)).flatten()

        obj = model.fit(X_train, y_train_s)
        y_pred_s = obj.predict(X_test)
        y_pred = scaler.inverse_transform(y_pred_s).flatten()
    else:
        obj = model.fit(X_train, y_train)
        y_pred = obj.predict(X_test)

    results = {}
    for score_name, score_fn in scores.items():
        results[score_name] = score_fn(y_test, y_pred)
    return results


def run_kfold(X, y, n_splits=10):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=12345)

    results = []
    for model_name, model in models.items():
        print(f"----- {model_name} -----")
        for i, (train_index, test_index) in enumerate(kf.split(X)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            temp_results = evaluate(model, X_train, y_train, X_test, y_test, scores, use_scaler=True)
            print(f"Fold = {i}: {temp_results}")

            temp_results['method'] = model_name
            temp_results['fold'] = i
            results.append(temp_results)
    
    return pd.DataFrame(results)


def create_single_model(model_name):
    model = models[model_name]
    obj = model.fit(X, y)

    filename = f'{model_name}.pickle'
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)
    print(f'Saved: {filename}')


if __name__ == "__main__":
    X, y = load_xy('main_data.xlsx')
    df = run_kfold(X, y, 10)

    cols = ['fold', 'method'] + list(scores.keys())
    df = df[cols]

    print('----- results.csv -----')
    print(df)
    df.to_csv('results.csv')

    # Summarize the results by taking the mean across all the k-folds
    print('----- summary.csv -----')
    df_summary = df.groupby('method').agg({
        'MSE': ['mean', 'std'],
        'MAE': ['mean', 'std'],
        'r2': ['mean', 'std']
    })
    print(df_summary)
    df_summary.to_csv('summary.csv')
    df_summary = df_summary.sort_values(('MSE', 'mean'), ascending=True)
    df_summary.to_excel('summary.xlsx')

    best_model = df_summary.index[0]
    create_single_model(best_model)
