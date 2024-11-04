import pandas as pd
import torch
from sklearn.svm import LinearSVR
import pickle
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import make_scorer
from sklearn.linear_model import Ridge
import numpy as np


def pearsonr_scorer(y_true, y_pred):
    return pearsonr(y_pred, y_true).correlation


with open('../processed_dataset_mistral_7b_instruct_v2_refelction.pkl', 'rb') as f:
    data = pickle.load(f)

X = np.array([[v for k, v in i.items() if k == 'embeddgins_15'] for i in data]).squeeze()
y = np.array([[v for k, v in i.items() if k == 'target'] for i in data]).squeeze()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

custom_scorer = make_scorer(pearsonr_scorer, greater_is_better=True)

from sklearn.metrics import r2_score
from sklearn.linear_model import BayesianRidge

br = BayesianRidge()
br.fit(X_train, y_train)

# Predicting values for the unseen data, i.e., the testing data
y_pred = br.predict(X_test)

# Computing the R-square score for the model
print(f"The r2 score of the model is: {r2_score(y_test, y_pred)}")
result = pearsonr(y_pred, y_test)
correlation = result.correlation
pvalue = result.pvalue
print("Test pearsonr:", correlation)
print("Test pvalue:", pvalue)
