import numpy as np
import pandas as pd
import torch
from sklearn.svm import LinearSVR
import pickle
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import make_scorer


def pearsonr_scorer(y_true, y_pred):
    return pearsonr(y_pred, y_true).correlation


with open('processed_dataset_mixtral_8x7B_instruct_qlora_nf4_reflect_forward.pkl', 'rb') as f:
    data = pickle.load(f)

X = np.array([[v for k, v in i.items() if k == 'embeddgins_17'] for i in data]).squeeze()
y = np.array([[v for k, v in i.items() if k == 'target'] for i in data]).squeeze()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

custom_scorer = make_scorer(pearsonr_scorer, greater_is_better=True)

model = LinearSVR(max_iter=100000, random_state=42)
param_grid = {
    'C': [0.1, 1, 10, 100, 0.5, 0.7, 0.3, 0.8, 2],
    'epsilon': [0.01, 0.1, 0.2],
    'dual': [True]
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(model, param_grid, cv=kf, scoring=custom_scorer, n_jobs=-1, verbose=2)

grid_search.fit(X_train, y_train)
print("Best hyperparameters:", grid_search.best_params_)
print("Best cross-validation pearsonr:", grid_search.best_score_)
