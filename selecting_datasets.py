from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV, ElasticNetCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sb
from numpy.linalg import inv

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import random

from pmlb import fetch_data, regression_dataset_names

list_dataset = []

for dataset in regression_dataset_names:
    X, y = fetch_data(dataset, return_X_y=True, local_cache_dir='/data2/yinterian/pmlb/')
    if X.shape[0] >= 1000:
        list_dataset.append(dataset)


alphas=[1e-4, 1e-3, 1e-2, 1e-1, 1, 2, 4, 8, 16, 32, 64, 132]
def other_scores(train_X, test_X, train_y, test_y):
    
    N = train_X.shape[1]
    max_depth = np.unique([int(x*N + 1) for x in np.linspace(0.01, 2, num = 5)])
    grid = {'max_depth': max_depth}
    rf = RandomForestRegressor(n_estimators=1000, max_features='sqrt', n_jobs = 10)
    rf_cv = GridSearchCV(estimator = rf, param_grid = grid, cv = 5, verbose=2,
                         n_jobs = 2)
    ridge  = RidgeCV(cv=5, alphas=alphas)
    lasso = ElasticNetCV(cv=5, random_state=0, l1_ratio=1)
    dt = DecisionTreeRegressor(min_samples_leaf=10)
    dt_cv = GridSearchCV(estimator = dt, param_grid = grid, cv = 5, verbose=2,
                         n_jobs = 20)
    
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.transform(test_X)
    
    rf_cv.fit(train_X, train_y)
    lasso.fit(train_X, train_y)
    ridge.fit(train_X, train_y)
    dt_cv.fit(train_X, train_y)
    scores = [x.score(test_X, test_y) for x in [rf_cv, ridge, lasso, dt_cv]]
    return scores

model_str = ["RF", "Ridge", "Lasso", "DT"]

f = open('comparison.log', 'w+')
for state in range(1,10):
    for dataset in list_dataset:
        X, y = fetch_data(dataset, return_X_y=True, local_cache_dir='/data2/yinterian/pmlb/')
        train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=state)
        scaler = StandardScaler()
        train_X = scaler.fit_transform(train_X)
        test_X = scaler.transform(test_X)
        scores = other_scores(train_X, test_X, train_y, test_y)
        score_str = ["%s %.4f" % (s, score) for s,score in zip(model_str, scores)]
        score_str = " ".join(score_str)
        results = "dataset %s state %d %s"  %(dataset, state, score_str)
        f.write(results)
        f.write("\n")
        f.flush()
f.close()
