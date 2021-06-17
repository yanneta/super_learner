from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sb
from numpy.linalg import inv

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import random


from pmlb import fetch_data, regression_dataset_names

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
    
    # L1 penalty
    def model_1(self):
        return LogisticRegressionCV(cv=5, penalty='l1',solver = 'saga', random_state=0)

    # l2 penalty
    def model_2(self):
        return LogisticRegressionCV(cv=5, penalty='l2', solver = 'saga', random_state=0)
    
    # elastic net
    def model_3(self):
        return LogisticRegressionCV(cv=5, penalty='elasticnet', solver = 'saga', l1_ratio=.5, random_state=0)

    def model_4(self):
        return DecisionTreeClassifier(max_depth=3)

    def model_5(self):
        return DecisionTreeClassifier(max_depth=4)

    def model_6(self):
        return DecisionTreeClassifier(max_depth=5)


def fit_initial_K_models(train_X, train_y, model_types):
    models = []
    N = train_X.shape[0]
    n = int(3*N/np.log(N))
    for k in range(len(model_types)):
        ind = np.random.choice(N, n, replace=False)
        X = train_X[ind]
        y = train_y[ind]
        if len(ind) > 10:
            base_model = BaseModel(model_types[k])
            base_model.model.fit(X, y)
            models.append(base_model)
    return models

def fit_K_models(train_X, train_y, oracle, models, p=0.8):
    # sample to address overfitting 
    N = train_X.shape[0]
    n = int(p*N)
    ind = np.random.choice(N, n, replace=False)
    X = train_X[ind]
    y = train_y[ind]
    # assigning points using oracle
    # this will be modified 
    x = torch.tensor(X).float()
    y_hat = oracle(x.cuda())
    W = F.softmax(0.5*y_hat, dim=1).cpu().detach().numpy()
                
    model_types = [m.model_type for m in models]
    models = []
    for k in range(len(model_types)):
        w = W[:,k]
        if w.sum()/n > 0.015:
            idx = w > 0.000001
            w = W[idx, k].copy() 
            X_k = X[idx]
            y_k = y[idx]
            base_model = BaseModel(model_types[k])
            print("model_type=", model_types[k], k)
            base_model.model.fit(X_k, y_k, w)
            models.append(base_model)
    return models


# use cross entropy loss
def compute_K_model_loss(train_X, train_y, models):
    L = []
    for i in range(len(models)):
        score = (models[i].model.score(train_X, train_y)
        L.append(score)
    L = np.array(L)
    return L

def compute_weights(L, K):
    JI_K = inv(np.ones((K, K)) - np.identity(K))
    W = []
    for i in range(L.shape[1]):
        w_i = np.matmul(JI_K, L[:,i])
        W.append(w_i)
    return np.array(W)

def create_extended_dataset(train_X, train_y, models, p=0.7):
    # sample to address overfitting
    K = len(models)
    N = train_X.shape[0]
    n = int(p*N)
    idx = np.random.choice(N, n, replace=False)
    X = train_X[idx]
    Y = train_y[idx]
    L = compute_K_model_loss(X, Y, models)
    W = compute_weights(L, K)
    X_ext = []
    y_ext = []
    w_ext = []
    for i in range(K):
        X_ext.append(X.copy())
        y_ext.append(i*np.ones(n))
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
    H = np.minimum(int(2*np.log(N)**2), 150) 
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
    #y_hat = beta*y_hat
    y_hat = F.softmax(y_hat, dim=1)
    loss = (y*(1-y_hat)).sum(dim=1)
    return (w*loss).mean()

def train_model(model, train_dl, K, learning_rate = 0.01, epochs=100):
    beta = 1
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.00001)
    KK = epochs//10 + 1
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
        if t % KK == 0: print("epoch %d loss %.4f" % (t, total_loss/total))

def assign_points(train_X, model):
    x = torch.tensor(train_X).float()
    y_hat = model(x.cuda())
    _, pred = torch.max(y_hat, 1)
    data = {'index': range(len(train_X)), 'group': pred.cpu().numpy()  }
    return pd.DataFrame(data) 


def relabel_groups(groups, models):
    unique_models = groups.group.unique()
    old2new = {x:i for i,x in enumerate(unique_models)}
    model_subset = [models[i] for i in unique_models]
    groups.group = np.array([old2new[x] for x in groups.group.values])
    return groups, model_subset


def compute_loss(X, y, oracle, models):
    oracle.eval()
    x = torch.tensor(X).float()
    y = torch.tensor(y).float()
    y_hat = oracle(x.cuda())
    _, ass = torch.max(y_hat, 1)
    preds = []
    ys = []
    k = 0
    for i in range(len(models)):
        xx = x[ass==i]
        yy = y[ass==i]
        if len(xx) > 0:
            k =+1
            pred = models[i].model.predict(xx.cpu().numpy())
            preds.append(pred)
            ys.append(yy.cpu().numpy())

    if k==1:
        preds, ys = preds[0], ys[0]
    else:
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


def baseline_models(train_X, train_y, valid_X, valid_y):
    best_model = None
    best_valid_r2 = 0
    best_model_type = 0
    for k in range(1,7):
        base_model = BaseModel(k)
        base_model.model.fit(train_X, train_y)
        valid_r2 = base_model.model.score(valid_X, valid_y)
        if valid_r2 > best_valid_r2:
            best_valid_r2 = valid_r2
            best_model_type = k
            best_model = base_model.model
    return best_valid_r2, best_model, [best_model_type]


#############################
# Main loop
############################
list_dataset = []
model_str = ["RF", "Ridge", "Lasso", "DT"]

#"1191_BNG_pbc", too expensive, leaving outside for now
# "1196_BNG_pharynx"
selected_datasets = ["1028_SWD", "1029_LEV", "1199_BNG_echoMonths", "1201_BNG_breastTumor",
        "1595_poker", "201_pol", "218_house_8L", "225_puma8NH", "294_satellite_image", "537_houses",
        "564_fried", "573_cpu_act", "574_house_16H", "1191_BNG_pbc", "1196_BNG_pharynx"]

selected_datasets = []
for dataset in regression_dataset_names:
    X, y = fetch_data(dataset, return_X_y=True, local_cache_dir='/data2/yinterian/pmlb/')
    if X.shape[0] >= 2000 and X.shape[0] <=500000:
        selected_datasets.append(dataset)

def get_datatest_split(dataset, state):
    X, y = fetch_data(dataset, return_X_y=True, local_cache_dir='/data2/yinterian/pmlb/')
    train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=state, test_size = 0.3)
    valid_X, test_X, valid_y, test_y = train_test_split(test_X, test_y, random_state=state, test_size =0.5)
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.transform(test_X)
    valid_X = scaler.transform(valid_X)
    return train_X, valid_X, test_X, train_y, valid_y, test_y


def main_loop(state):
    for dataset in selected_datasets:
        learning_rate = 0.15
        train_X, valid_X, test_X, train_y, valid_y, test_y = get_datatest_split(dataset, state)

        best_valid_r2, best_model, best_model_types = baseline_models(train_X, train_y, valid_X, valid_y)
        best_test_r2 = best_model.score(test_X, test_y)
        best_single_model = best_test_r2 
        print("best valid R^2 %.3f best model type %d" % (best_valid_r2, best_model_types[0]))
        best_oracle = None
        best_models = [best_model] 

        batch_size = 100000
        # number of iterations depends on the number of training points
        N = train_X.shape[0]
        N_iter = int(3000/np.log(N)**2)
        print("Number of training points %d, number iterations %d" % (N, N_iter))

        model_types = [x for x in range(1,7)]
        K = len(model_types)
        INIT_FLAG = True
        for i in range(16):
            if i == 7: INIT_FLAG = True
            
            if not INIT_FLAG:
                models = fit_K_models(train_X, train_y, oracle, models, p=0.9)
                if len(models) == 1:
                    INIT_FLAG = True  
            
            if INIT_FLAG:
                #model_types = [x for x in range(1,7)] + [1,3,6,6,6,6,6,6]
                model_types = [1,1,1,4,5,6,6,6,6,6]
                models = fit_initial_K_models(train_X, train_y, model_types)
                INIT_FLAG = False
            
            K = len(models)
            print("Iteration %d K is %d" % (i+1, K))
            if K == 1:
                INIT_FLAG = True

            if not INIT_FLAG:
                X_ext, y_ext, w_ext = create_extended_dataset(train_X, train_y, models, p=0.7)
                train_ds = OracleDataset(X_ext, y_ext, w_ext)
                train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
                oracle = create_oracle_model(train_X.shape[1], K, N).cuda()
                train_model(oracle, train_dl, K, learning_rate, N_iter)
            
            if not INIT_FLAG:
                train_loss, train_r2 = compute_loss(train_X, train_y, oracle, models)
                valid_loss, valid_r2 = compute_loss(valid_X, valid_y, oracle, models)
                test_loss, test_r2 = compute_loss(test_X, test_y, oracle, models)

            print("train loss %.3f valid loss %.3f", train_loss, valid_loss)
            print("train R^2 %.3f valid R^2 %.3f", train_r2, valid_r2)
            if valid_r2 >= best_valid_r2:
                best_train_r2 = train_r2
                best_valid_r2 = valid_r2
                best_K = K
                best_models = models
                best_model_types = [m.model_type for m in models]
                best_test_r2 = test_r2 
        
        results = "dataset %s state %d K %d test ISL %.3f valid ISL %.3f test base model %.3f model_types %s" % (
                dataset, state, len(best_models), best_test_r2, best_valid_r2, best_single_model, str(best_model_types))
        print(results)
        f.write(results)
        f.write('\n')
        f.flush()

f = open('results-july-11-2019-weight-dacay.log', 'w+')
for state in range(1, 11):
    main_loop(state)
f.close()
