import pickle
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from sklearn.metrics import make_scorer
from sklearn.linear_model import RidgeCV
import numpy as np


def pearsonr_scorer(y_true, y_pred):
    return pearsonr(y_true, y_pred).correlation


with open('processed_dataset_mixtral_8x7B_instruct_qlora_nf4_reflect_forward.pkl', 'rb') as f:
    data = pickle.load(f)

X = np.array([[v for k, v in i.items() if k == 'embeddgins_17'] for i in data]).squeeze()
y = np.array([[v for k, v in i.items() if k == 'target'] for i in data])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

custom_scorer = make_scorer(pearsonr_scorer, greater_is_better=True)

model = RidgeCV(alphas=[10 ** x for x in range(-30, 30)], scoring=custom_scorer, alpha_per_target=True)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
correlation, pvalue = pearsonr(y_test.squeeze(), y_pred.squeeze())

print("Best alpha:", model.alpha_)
print("Test Pearson correlation:", correlation)
print("Test p-value:", pvalue)
