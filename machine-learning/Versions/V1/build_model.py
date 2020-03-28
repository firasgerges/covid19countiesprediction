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


warnings.filterwarnings('ignore')

models = {
    'dtr': DecisionTreeRegressor(),
    'linear': linear_model.LinearRegression(),
    'logistic': linear_model.LogisticRegression(),
    'knr': KNeighborsRegressor(),
    'svr': svm.SVR(),
    'gbr': GradientBoostingRegressor(),
    'rf': RandomForestRegressor(n_estimators=100, criterion='mse')
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


def evaluate(model, X_train, y_train, X_test, y_test, scores):
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
            temp_results = evaluate(model, X_train, y_train, X_test, y_test, scores)
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
    X, y = load_xy('../main_data.xlsx')
    df = run_kfold(X, y, 10)

    cols = ['fold', 'method'] + list(scores.keys())
    df = df[cols]

    print('----- results.csv -----')
    print(df)
    df.to_csv('Results/results.csv')

    # Summarize the results by taking the mean across all the k-folds
    print('----- summary.csv -----')
    df_summary = df.groupby('method').agg({'MSE': 'mean', 'MAE': 'mean', 'r2': 'mean'})
    print(df_summary)
    df_summary.to_csv('Results/summary.csv')
    df_summary = df_summary.sort_values('MSE', ascending=True)
    df_summary.to_excel('Results/summary.xlsx')

    best_model = df_summary.index[0]
    create_single_model(best_model)
