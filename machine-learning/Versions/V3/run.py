#!/usr/bin/env python

# This script optimizes the model

import hyperopt
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import warnings

from functools import partial
from sklearn.model_selection import KFold, train_test_split
from sklearn import datasets, linear_model, svm
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
warnings.filterwarnings('ignore')

def load_xy(filename):
    df = pd.read_excel(filename, index_col=0)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X.values, y.values

def evaluate(model, X_train, y_train, X_test, y_test):
    """
    Returns the mean squared error of the model, as evaluated against the test set
    `_s` suffix is the Standard Scaler version of the same name
    """
    #lab_enc = preprocessing.LabelEncoder()
    #encoded = lab_enc.fit_transform(y_train)
    obj = model.fit(X_train, y_train)
    y_pred = obj.predict(X_test)
    print( mean_absolute_error(y_test, y_pred));
    print( r2_score(y_test, y_pred));
    return y_pred



def check_optimized(model, X, y):
    """
    Build model with the optimized parameters, make a prediction,
    and compare the results
    """
    X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=0.2, random_state=12345)
    y_pred = evaluate(model, X_train, y_train, X_test, y_test)
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    ax.plot(y_test, y_pred, 'x')
    ax.plot(y_test, y_test, '--')
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predictied')
    plt.show()
    df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred
    })
    return df

models = {
    'dtr': DecisionTreeRegressor(criterion='mse',splitter='best',max_features=1.0),
    'linear': linear_model.LinearRegression(),
    'sgd': linear_model.SGDRegressor(),
    'gnb': GaussianNB(),
    'lars': linear_model.LassoLars(alpha=.1),
    'logistic': linear_model.LogisticRegression(),
    'knr': KNeighborsRegressor(weights='distance'),
    'svr': svm.SVR(),
    'svr_rbf': svm.SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1),
    'svr_lin': svm.SVR(kernel='linear', C=100, gamma='auto'),
    'svr_poly': svm.SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1, coef0=1),
    'gbr': GradientBoostingRegressor(),
    'rf': RandomForestRegressor(n_estimators=100, criterion='mse')
}
model_name='dtr'
run_model=models[model_name];
if __name__ == "__main__":
    X, y = load_xy('main_data.xlsx')
    df_check = check_optimized(run_model, X, y)
    print(df_check)
    obj = run_model.fit(X, y)
    filename = f'{model_name}.sav'
    with open(filename, 'wb') as f:
        pickle.dump(obj, f,protocol=2)