import pandas as pd
import torch
from sklearn.svm import LinearSVR
import pickle
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import make_scorer
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor


def pearsonr_scorer(y_true, y_pred):
    return pearsonr(y_pred, y_true).correlation


with open('../processed_dataset_mistral_7b_instruct_v2_refelction.pkl', 'rb') as f:
    data = pickle.load(f)

with open('../processed_dataset_llama3_8b.pkl', 'rb') as f:
    data_generate = pickle.load(f)

features = sorted(list(set(data[0].keys()) - {'target'}))
mx = -2
all_y_preds = []

for feature in features:
    X = np.array([[v for k, v in i.items() if k == feature] for i in data]).squeeze()
    y = np.array([[v for k, v in i.items() if k == 'target'] for i in data]).squeeze()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    custom_scorer = make_scorer(pearsonr_scorer, greater_is_better=True)

    # Create a support vector regression model
    model = LinearSVR(max_iter=10000, random_state=42, dual=True,
                      # **{'C': 0.7, 'dual': True, 'epsilon': 0.2}
                      )
    # model = Ridge(max_iter=10000, random_state=42,
    #               **{'alpha': 0.1, 'fit_intercept': False, 'solver': 'lsqr', 'tol': 0.0001}
    #               )

    # Step 6: Fit the model to the data
    model.fit(X_train, y_train)
    # Step 7: Get the best parameters and the corresponding score
    y_pred = model.predict(X_test)
    all_y_preds.append(y_pred)
    result = pearsonr(y_pred, y_test)
    correlation = result.correlation
    pvalue = result.pvalue
    if mx < correlation:
        mx = correlation
        best_feature = feature
    print('Feature:', feature)
    print("Test pearsonr:", correlation)
    print("Test pvalue:", pvalue)

features_generate = sorted(list(set(data_generate[0].keys()) - {'target'}))

for feature in features_generate:
    X = np.array([[v for k, v in i.items() if k == feature] for i in data]).squeeze()
    y = np.array([[v for k, v in i.items() if k == 'target'] for i in data]).squeeze()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    custom_scorer = make_scorer(pearsonr_scorer, greater_is_better=True)

    # Create a support vector regression model
    model = LinearSVR(max_iter=10000, random_state=42, dual=True,
                      # **{'C': 0.7, 'dual': True, 'epsilon': 0.2}
                      )
    # model = Ridge(max_iter=10000, random_state=42,
    #               **{'alpha': 0.1, 'fit_intercept': False, 'solver': 'lsqr', 'tol': 0.0001}
    #               )

    # Step 6: Fit the model to the data
    model.fit(X_train, y_train)
    # Step 7: Get the best parameters and the corresponding score
    y_pred = model.predict(X_test)
    all_y_preds.append(y_pred)
    result = pearsonr(y_pred, y_test)
    correlation = result.correlation
    pvalue = result.pvalue
    if mx < correlation:
        mx = correlation
        best_feature = feature
    print('Feature:', feature)
    print("Test pearsonr:", correlation)
    print("Test pvalue:", pvalue)

avg_y_pred = np.array(all_y_preds).mean(axis=0)
result_avg = pearsonr(avg_y_pred, y_test)
correlation_avg = result_avg.correlation
pvalue_avg = result_avg.pvalue
print('Average pearsonr:', correlation_avg)
print('Average pvalue:', pvalue_avg)

print('Best feature:', best_feature)
print('Best pearsonr:', mx)


