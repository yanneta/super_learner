#!/usr/bin/env python
# coding: utf-8

from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegressionCV

import pandas as pd
import numpy as np
from pathlib import Path
import random


import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from pmlb import fetch_data, classification_dataset_names


alphas=[1e-4, 1e-3, 1e-2, 1e-1, 1, 2, 4, 8, 16, 32, 64, 132]
class BaseModel:
    def __init__(self, model_type):
        self.model_type = model_type
        self.model = self.create_model()
        if model_type not in range(1,7):
            print("model_type should be in the interval [1, 6]")

    def create_model(self):
        method_name = 'model_' + str(self.model_type)
        method = getattr(self, method_name, lambda: "nothing")
        return method()

    # L1 penalty
    def model_1(self):
        return LogisticRegressionCV(cv=5, penalty='l1',solver = 'saga', random_state=0)

    # l2 penalty
    def model_2(self):
        return LogisticRegressionCV(cv=5, penalty='l2', solver = 'saga', random_state=0)

    # elastic net
    def model_3(self):
        return LogisticRegressionCV(cv=5, penalty='elasticnet', solver = 'saga', l1_ratios=[.5], random_state=0)

    def model_4(self):
        return DecisionTreeClassifier(max_depth=3)

    def model_5(self):
        return DecisionTreeClassifier(max_depth=4)

    def model_6(self):
        return DecisionTreeClassifier(max_depth=5)



def random_assignments(train_X, K=5):
    data = {'index': range(len(train_X)), 'group':  np.random.choice(K, len(train_X)) }
    df = pd.DataFrame(data)
    return df


alphas=[1e-4, 1e-3, 1e-2, 1e-1, 1, 2, 4, 8, 16, 32, 64, 132]
def fit_models(X, y, model_types):
    base_models = []
    for j in model_types:
        base_model = BaseModel(j)
        base_model.model.fit(X, y)
        base_models.append(base_model)
    return base_models


def train_stacking_models(train_X, train_y, model_types, groups, K=5):
    models = {}
    # k fold
    for k in range(K):
        ind = groups[groups["group"] != k].index.values
        X = train_X[ind]
        y = train_y[ind]
        models[k] = fit_models(X, y, model_types)
    return models



def create_first_layer_preds(X, base_models):
    preds = []
    for j, model in enumerate(base_models):
        pred = model.model.predict_proba(X)
        preds.append(pred)
    return np.concatenate(preds, axis=1)


def create_stacking_dataset(train_X, train_y, models, groups, K=5):
    pred_list = []
    y_list = []
    for k in range(K):
        ind = groups[groups["group"] == k].index.values
        X = train_X[ind]
        y = train_y[ind]
        preds = create_first_layer_preds(X, models[k])
        pred_list.append(preds)
        y_list.append(y)
    return np.concatenate(pred_list), np.concatenate(y_list)


def fit_stacking_model(train_X, train_y, model_types=[1,2,3,4,5,6], K=5):
    groups = random_assignments(train_X, K)
    models = train_stacking_models(train_X, train_y, model_types, groups, K)
    X, y = create_stacking_dataset(train_X, train_y, models, groups, K)
    s_model = LogisticRegressionCV(cv=5, penalty='l1',solver = 'saga', random_state=0)
    s_model.fit(X, y)
    base_models = fit_models(train_X, train_y, model_types)
    return base_models, s_model


def acc_stack_model(test_X, test_y, base_models, s_model):
    X = create_first_layer_preds(test_X, base_models) 
    pred = s_model.predict(X)
    acc = (pred == test_y).sum()/test_y.shape[0]
    return acc


def get_class_data(dataset):
    X, y = fetch_data(dataset, return_X_y=True, local_cache_dir='/data2/yinterian/pmlb/')
    y_min = np.unique(y).min()
    if y_min == 1:
        y -= 1
    return X, y

def get_datatest_split(dataset, state):
    X, y = get_class_data(dataset)
    train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=state, test_size = 0.3)
    valid_X, test_X, valid_y, test_y = train_test_split(test_X, test_y, random_state=state, test_size =0.5)
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.transform(test_X)
    valid_X = scaler.transform(valid_X)
    return train_X, valid_X, test_X, train_y, valid_y, test_y


list_dataset = [ 'agaricus_lepiota', 'churn', 'connect_4', 'krkopt', 'phoneme', 'ring']

def main_loop(state, dataset):
    train_X, valid_X, test_X, train_y, valid_y, test_y = get_datatest_split(dataset, state)
    base_models, s_model = fit_stacking_model(train_X, train_y)
    test_acc = acc_stack_model(test_X, test_y, base_models, s_model)
    results = "dataset %s state %d test stacking %.3f" % (
            dataset, state, test_acc)
    print(results)
    f.write(results)
    f.write('\n')
    f.flush()


list_dataset = [ 'agaricus_lepiota', 'churn', 'connect_4']
#list_dataset = ['krkopt', 'phoneme', 'ring']
f = open('test1.log', 'w+')
for dataset in list_dataset:
    for state in range(1, 11):
        main_loop(state, dataset)
f.close()


