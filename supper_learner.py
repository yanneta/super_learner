from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score
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


def other_scores(train_X, test_X, train_y, test_y):
    alphas=[1e-4, 1e-3, 1e-2, 1e-1, 1, 2, 4, 8, 16, 32, 64]
    
    rf_reg = RandomForestRegressor(n_estimators=10, max_depth=15, n_jobs=10)
    ridge_reg = RidgeCV(alphas=alphas)
    lasso_reg = LassoCV(cv=5, random_state=0)

    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.transform(test_X)
    
    rf_reg.fit(train_X, train_y)
    lasso_reg.fit(train_X, train_y)
    ridge_reg.fit(train_X, train_y)

    return rf_reg.score(test_X, test_y), ridge_reg.score(test_X, test_y), lasso_reg.score(test_X, test_y)


def random_assignments(train_X, K=6):
    data = {'index': range(len(train_X)), 'group':  np.random.choice(K, len(train_X)) }
    df = pd.DataFrame(data)
    return df

def fit_K_models(train_X, train_y, groups, alpha, K=6):
    models = []
    for k in range(K):
        ind = groups[groups["group"] == k].index.values
        X = train_X[ind]
        y = train_y[ind]
        ridge_reg = Ridge(alpha=alpha)
        ridge_reg.fit(X, y)
        models.append(ridge_reg)
    return models

def compute_K_model_loss(train_X, train_y, models):
    L = []
    for i in range(len(models)):
        loss = (models[i].predict(train_X) - train_y)**2
        L.append(loss)
    L = np.array(L)
    return L


def compute_weights(L, K):
    JI_K = inv(np.ones((K, K)) - np.identity(K))
    W = []
    for i in range(L.shape[1]):
        w_i = np.matmul(JI_K, L[:,i])
        W.append(w_i)
    return np.array(W)

def create_extended_dataset(train_X, train_y, models):
    K = len(models)
    N = train_X.shape[0]
    L = compute_K_model_loss(train_X, train_y, models)
    W = compute_weights(L, K)
    X_ext = []
    y_ext = []
    w_ext = []
    for i in range(K):
        X_ext.append(train_X.copy())
        y_ext.append(i*np.ones(N))
        w_ext.append(W[:, i])
    X_ext = np.concatenate(X_ext, axis=0)
    y_ext = np.concatenate(y_ext, axis=0)
    w_ext = np.concatenate(w_ext, axis=0)
    return X_ext, y_ext, w_ext

def create_model(D_in, K, H=512):
    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        nn.BatchNorm1d(H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, K))
    return model

def softmax_loss(beta, f_hat, y, w):
    y_hat = np.exp(beta*f_hat)
    den = (np.exp(beta*f_hat)).sum(axis=1)
    y_hat = np.array([y_hat[i]/den[i] for i in range(len(den))])
    loss = w*((y * (1- y_hat)).sum(axis=1))
    return loss.mean()

def bounded_loss(beta, y_hat, y , w):
    y_hat = beta*y_hat
    y_hat = F.softmax(y_hat, dim=1)
    loss = (y*(1-y_hat)).sum(dim=1)
    return (w*loss).mean()

def train_model(model, train_dl, K, learning_rate = 0.01, epochs=100):
    beta = 1
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    KK = epochs//10
    for t in range(epochs):
        total_loss = 0
        total = 0
        for x, y, w in train_dl:
            x = x.cuda().float()
            y = y.cuda().long()
            w = w.cuda().float()
            y_onehot = torch.FloatTensor(y.shape[0], K).cuda()
            y_onehot.zero_()
            y_onehot = y_onehot.scatter_(1, y.unsqueeze(1), 1)
            y_hat = model(x)
            loss = bounded_loss(beta, y_hat, y_onehot , w)
       
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()*y.size(0)
            total += y.size(0)
        if t % KK == 0: print(total_loss/total)

def reasign_points(train_X, model):
    x = torch.tensor(train_X).float()
    y_hat = model(x.cuda())
    _, pred = torch.max(y_hat, 1)
    data = {'index': range(len(train_X)), 'group': pred.cpu().numpy()  }
    return pd.DataFrame(data) 

def relabel_groups(groups):
    old2new = {x:i for i,x in enumerate(groups.group.unique())}
    groups.group = np.array([old2new[x] for x in groups.group.values])
    return groups

def compute_loss(X, y, oracle, models):
    x = torch.tensor(X).float()
    y = torch.tensor(y).float()
    y_hat = oracle(x.cuda())
    _, ass = torch.max(y_hat, 1)
    preds = []
    ys = []
    for i in range(len(models)):
        xx = x[ass==i]
        yy = y[ass==i]
        if len(xx) > 0:
            pred = models[i].predict(xx.cpu().numpy())
            preds.append(pred)
            ys.append(yy.cpu().numpy())
    preds = np.hstack(preds)
    ys = np.hstack(ys)
    r2 = r2_score(ys, preds)
    res = (ys - preds)**2
    return res.mean(), r2


class OracleDataset(Dataset):
    def __init__(self, X, y, w):
        self.X = X
        self.y = y
        self.w = w
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.w[idx]


#############################
# Main loop
############################
alphas=[1e-4, 1e-3, 1e-2, 1e-1, 1, 2, 4, 8, 16, 32, 64]
Hidden=100
list_dataset = []
learning_rate = 0.01

for dataset in regression_dataset_names:
    X, y = fetch_data(dataset, return_X_y=True, local_cache_dir='/data2/yinterian/pmlb/')
    if X.shape[0] >= 1000:
        list_dataset.append(dataset)
f = open('out.log', 'w+')
for dataset in list_dataset:
    X, y = fetch_data(dataset, return_X_y=True, local_cache_dir='/data2/yinterian/pmlb/')
    train_X, test_X, train_y, test_y = train_test_split(X, y)
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.transform(test_X)

    ridge_reg = RidgeCV(alphas=alphas)
    ridge_reg.fit(train_X, train_y)
    alpha = ridge_reg.alpha_

    print(alpha) 
    K = 6
    groups = random_assignments(train_X, K)

    batch_size = 100000
    # number of iterations depends on the number of training points
    N = train_X.shape[0]
    N_iter = int(10000/np.log(N)**2)
    print("Number of training points %d, number iterations %d" % (N, N_iter))

    best_train_r2 = None
    best_K = None
    best_test_r2 = None
    for i in range(10):
        train_loss = None
        print("Iteration %d K is %d" % (i+1, K))
        models = fit_K_models(train_X, train_y, groups, alpha, K)
        X_ext, y_ext, w_ext = create_extended_dataset(train_X, train_y, models)
        train_ds = OracleDataset(X_ext, y_ext, w_ext)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        model = create_model(train_X.shape[1], K, H=Hidden).cuda()
        train_model(model, train_dl, K, learning_rate, N_iter)
        groups = reasign_points(train_X, model)
        if len(groups.group.unique()) < K:
            K = len(groups.group.unique()) 
            groups = relabel_groups(groups)
        train_loss, train_r2 = compute_loss(train_X, train_y, model, models)
        test_loss, test_r2 = compute_loss(test_X, test_y, model, models)
        print("loss", train_loss, test_loss)
        print("R^2", train_r2, test_r2)
        if best_train_r2 == None:
            best_train_r2 = train_r2
        if train_r2 >= best_train_r2:
            best_train_r2 = train_r2
            best_test_r2 = test_r2
            best_K = K
        if K == 1:
            print("K", K)
            break

    r2_rf, r2_ridge, r2_lasso = other_scores(train_X, test_X, train_y, test_y)
    results = "dataset %s K %d ISL_r^2 %.4f RF_r^2 %.4f Ridge_r^2 %.4f Lasso_r^2 %.4f" %(dataset, best_K, best_test_r2, r2_rf, r2_ridge, r2_lasso)
    print(results)
    f.write(results)
    f.write("\n")
    f.flush()
f.close()
