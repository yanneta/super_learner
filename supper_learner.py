from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV, ElasticNetCV
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


alphas=[1e-4, 1e-3, 1e-2, 1e-1, 1, 2, 4, 8, 16, 32, 64, 132]
def other_scores(train_X, test_X, train_y, test_y):
    rf = RandomForestRegressor(n_estimators=10, max_depth=15, n_jobs=10)
    ridge  = RidgeCV(cv=5, alphas=alphas)
    lasso = ElasticNetCV(cv=5, random_state=0, l1_ratio=1)
    dt = DecisionTreeRegressor(min_samples_leaf=10)
    
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.transform(test_X)
    
    rf.fit(train_X, train_y)
    lasso.fit(train_X, train_y)
    ridge.fit(train_X, train_y)
    dt.fit(train_X, train_y)
    scores = [x.score(test_X, test_y) for x in [rf, ridge, lasso, dt]]
    return scores

def random_assignments(train_X, K=6):
    data = {'index': range(len(train_X)), 'group':  np.random.choice(K, len(train_X)) }
    df = pd.DataFrame(data)
    return df


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

    def model_1(self):
        return RidgeCV(cv=5, alphas=alphas)

    def model_2(self):
        return ElasticNetCV(cv=5, random_state=0, l1_ratio=0.5)

    def model_3(self):
        return ElasticNetCV(cv=5, random_state=0, l1_ratio=1)

    def model_4(self):
        return DecisionTreeRegressor(max_depth=1)

    def model_5(self):
        return DecisionTreeRegressor(max_depth=3)

    def model_6(self):
        return DecisionTreeRegressor(max_depth=5)


def fit_initial_K_models(train_X, train_y, model_types):
    models = []
    N = train_X.shape[0]
    n = int(2.5*N/np.log(N))
    for k in range(len(model_types)):
        ind = np.random.choice(N, n, replace=False)
        X = train_X[ind]
        y = train_y[ind]
        if len(ind) > 10:
            base_model = BaseModel(model_types[k])
            base_model.model.fit(X, y)
            models.append(base_model)
    return models

def fit_K_models(train_X, train_y, groups, model_types, K=6):
    models = []
    for k in range(K):
        ind = groups[groups["group"] == k].index.values
        X = train_X[ind]
        y = train_y[ind]
        if len(ind) > 10:
            base_model = BaseModel(model_types[k])
            base_model.model.fit(X, y)
            models.append(base_model)
    return models

def compute_K_model_loss(train_X, train_y, models):
    L = []
    for i in range(len(models)):
        loss = (models[i].model.predict(train_X) - train_y)**2
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

def create_oracle_model(D_in, K, N):
    """ Returns an oracle model
    
    The size of the hidden layer is a function of the
    amount of training data
    """
    
    H = int(2*np.log(N)**2)
    model = nn.Sequential(
        nn.Linear(D_in, H),
        nn.BatchNorm1d(H),
        nn.ReLU(),
        nn.Dropout(p=0.2),
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
    model.train()
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


def relabel_groups(groups, models):
    unique_models = groups.group.unique()
    old2new = {x:i for i,x in enumerate(unique_models)}
    ratios = []
    model_types = [models[i].model_type for i in unique_models]
    groups.group = np.array([old2new[x] for x in groups.group.values])
    return groups, model_types


def compute_loss(X, y, oracle, models):
    oracle.eval()
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
            pred = models[i].model.predict(xx.cpu().numpy())
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


def compute_single_loss(X, y, model):
    pred = model.model.predict(X)
    r2 = r2_score(y, pred)
    res = (y - pred)**2
    return res.mean(), r2

#############################
# Main loop
############################
list_dataset = []
model_str = ["RF", "Ridge", "Lasso", "DT"]


lr_map = {"1028_SWD": 0.15, "1029_LEV" :0.15, "1030_ERA": 0.15, "1191_BNG_pbc": 0.02,
         "1193_BNG_lowbwt": 0.1, "1196_BNG_pharynx": 0.015, "1199_BNG_echoMonths": 0.3,
         "1203_BNG_pwLinear": 0.05, "1595_poker": 0.01, "1201_BNG_breastTumor": 0.05, "197_cpu_act": 0.2,
         "201_pol": 0.15, "215_2dplanes": 0.1, "218_house_8L": 0.05, "225_puma8NH": 0.15,
         "227_cpu_small":0.15, "294_satellite_image": 0.15, "344_mv": 0.1,
          "4544_GeographicalOriginalofMusic": 0.15, "503_wind": 0.1, "529_pollen": 0.1,
         "537_houses": 0.15, "562_cpu_small": 0.15, "564_fried": 0.1, "573_cpu_act": 0.15,
         "574_house_16H": 0.15}

selected_datasets = ["1028_SWD", "1191_BNG_pbc", "1196_BNG_pharynx", "1199_BNG_echoMonths", "1201_BNG_breastTumor",
        "1595_poker", "201_pol", "218_house_8L", "225_puma8NH", "294_satellite_image", "537_houses",
        "564_fried", "573_cpu_act", "574_house_16H"]


f = open('out.log', 'w+')
state = 1
for dataset in selected_datasets:
    learning_rate = lr_map.get(dataset, 0.15)
    X, y = fetch_data(dataset, return_X_y=True, local_cache_dir='/data2/yinterian/pmlb/')
    train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=state)
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.transform(test_X)

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
    best_model_types = None
    model_types = [x for x in range(1,7)]
    model_types = model_types + model_types
    K = len(model_types)
    for i in range(10):
        train_loss = None
        print("Iteration %d K is %d" % (i+1, K))
        if i == 0:
            models = fit_initial_K_models(train_X, train_y, model_types)
        else:
            models = fit_K_models(train_X, train_y, groups, model_types, K)
        K = len(models)
        if K == 1:
            models[0].model.fit(train_X, train_y)
            train_loss, train_r2 = compute_single_loss(train_X, train_y, models[0])
            test_loss, test_r2 = compute_single_loss(test_X, test_y, models[0])
            if train_r2 >= best_train_r2:
                best_train_r2 = train_r2
                best_test_r2 = test_r2
                best_K = K
                best_model_types = model_types
            break
        X_ext, y_ext, w_ext = create_extended_dataset(train_X, train_y, models)
        train_ds = OracleDataset(X_ext, y_ext, w_ext)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        model = create_oracle_model(train_X.shape[1], K, N).cuda()
        train_model(model, train_dl, K, learning_rate, N_iter)
        groups = reasign_points(train_X, model)
        if len(groups.group.unique()) < K:
            groups, model_types = relabel_groups(groups, models)
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
            best_model_types = model_types
   
    model_types_str = " ".join([str(x) for x in best_model_types])
    results = "dataset %s state %d ISL %.4f K %d model_types %s"  %(dataset, state, best_test_r2, best_K, model_types_str)
    print(results)
    f.write(results)
    f.write("\n")
    f.flush()
f.close()
