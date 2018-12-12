from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV, ElasticNetCV
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sb
from numpy.linalg import inv

import pandas as pd
import numpy as np
from pathlib import Path
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

    def model_1(self):
        return RidgeCV(cv=5, alphas=alphas)

    def model_2(self):
        return ElasticNetCV(cv=5, random_state=0, l1_ratio=0.5)

    def model_3(self):
        return ElasticNetCV(cv=5, random_state=0, l1_ratio=1)

    def model_4(self):
        return DecisionTreeRegressor(max_depth=3)

    def model_5(self):
        return DecisionTreeRegressor(max_depth=4)

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


def find_optimal_max_depth(X, y):
    N = X.shape[1]
    max_depth = np.unique([int(x*N + 1) for x in np.linspace(0.01, 2, num = 5)])
    grid = {'max_depth': max_depth}
    dt = DecisionTreeRegressor(min_samples_leaf=10)
    dt_cv = GridSearchCV(estimator = dt, param_grid = grid, cv = 5, verbose=2,
                         n_jobs = 20)
    dt_cv.fit(X, y)
    return dt_cv.best_params_["max_depth"]


def fit_tree_oracle_model(X, y, w, max_depth):
    dtc = DecisionTreeClassifier(max_depth=max_depth)
    dtc.fit(X, y, sample_weight=w)
    return dtc

def reasign_points(X, model):
    y_hat = model.predict(X).astype(int)
    data = {'index': range(len(X)), 'group': y_hat}
    return pd.DataFrame(data)


def relabel_groups(groups, models):
    unique_models = groups.group.unique()
    old2new = {x:i for i,x in enumerate(unique_models)}
    ratios = []
    model_types = [models[i].model_type for i in unique_models]
    groups.group = np.array([old2new[x] for x in groups.group.values])
    return groups, model_types


def compute_loss(X, y, oracle, models):
    y_hat = oracle.predict(X)
    preds = []
    ys = []
    for i in range(len(models)):
        xx = X[y_hat==i]
        yy = y[y_hat==i]
        if len(xx) > 0:
            pred = models[i].model.predict(xx)
            preds.append(pred)
            ys.append(yy)
    preds = np.hstack(preds)
    ys = np.hstack(ys)
    r2 = r2_score(ys, preds)
    res = (ys - preds)**2
    return res.mean(), r2

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


def relabel_groups(groups, models):
    unique_models = groups.group.unique()
    old2new = {x:i for i,x in enumerate(unique_models)}
    model_subset = [models[i] for i in unique_models]
    groups.group = np.array([old2new[x] for x in groups.group.values])
    return groups, model_subset, [m.model_type for m in model_subset]


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


def main_loop(state):
    for dataset in selected_datasets:
        X, y = fetch_data(dataset, return_X_y=True, local_cache_dir='/data2/yinterian/pmlb/')
        train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=state, test_size = 0.2)
        valid_X, test_X, valid_y, test_y = train_test_split(test_X, test_y, random_state=state, test_size =0.5)
        scaler = StandardScaler()
        train_X = scaler.fit_transform(train_X)
        test_X = scaler.transform(test_X)
        valid_X = scaler.transform(valid_X)
        max_depth = find_optimal_max_depth(train_X, train_y)

        best_valid_r2, best_model, best_model_types = baseline_models(train_X, train_y, valid_X, valid_y)
        best_test_r2 = best_model.score(test_X, test_y)
        print("best valid R^2 %.3f best model type %d" % (best_valid_r2, best_model_types[0]))
        best_oracle = None
        best_models = [best_model] 

        K = 6
        groups = random_assignments(train_X, K)

        batch_size = 100000
        # number of iterations depends on the number of training points
        N = train_X.shape[0]
        N_iter = int(3000/np.log(N)**2)
        print("Number of training points %d, number iterations %d" % (N, N_iter))

        model_types = [x for x in range(1,7)]
        model_types = model_types + [1,2,3,6,6,6]
        K = len(model_types)
        INIT_FLAG = True
        for i in range(16):
            if i == 8: INIT_FLAG = True
            print("Iteration %d K is %d" % (i+1, K))
            if INIT_FLAG:
                model_types = [x for x in range(1,7)] + [1,3,6,6,6]
                models = fit_initial_K_models(train_X, train_y, model_types)
                INIT_FLAG = False
            else:
                models = fit_K_models(train_X, train_y, groups, model_types, K)
            K = len(models)
            if K == 1:
                INIT_FLAG = True

            if not INIT_FLAG:
                X_ext, y_ext, w_ext = create_extended_dataset(train_X, train_y, models)
                oracle = fit_tree_oracle_model(X_ext, y_ext, w_ext, max_depth)
            
            groups = reasign_points(train_X, oracle)
            if len(groups.group.unique()) == 1:
                INIT_FLAG = True
            
            if not INIT_FLAG:
                train_loss, train_r2 = compute_loss(train_X, train_y, oracle, models)
                valid_loss, valid_r2 = compute_loss(valid_X, valid_y, oracle, models)
                test_loss, test_r2 = compute_loss(test_X, test_y, oracle, models)

            if len(groups.group.unique()) < K:
                groups, models, model_types = relabel_groups(groups, models)
                K = len(groups.group.unique())
                print(len(models), len(model_types), K)

            print("train loss %.3f valid loss %.3f", train_loss, valid_loss)
            print("train R^2 %.3f valid R^2 %.3f", train_r2, valid_r2)
            if valid_r2 >= best_valid_r2:
                best_train_r2 = train_r2
                best_valid_r2 = valid_r2
                best_K = K
                best_models = models
                best_model_types = model_types
                best_test_r2 = test_r2 
        
        results = "dataset %s state %d K %d test ISL %.3f valid ISL %.3f model_types %s" % (
                dataset, state, len(best_models), best_test_r2, best_valid_r2, str(best_model_types))
        print(results)
        f.write(results)
        f.write('\n')
        f.flush()

f = open('out_tree.log', 'w+')
for state in range(1, 11):
    main_loop(state)
f.close()
