import pickle
import numpy as np
from sklearn.model_selection import KFold
import torch.nn as nn
from scipy.stats import pearsonr
import torch
from torch.utils.data import Dataset, DataLoader
import copy


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(MLP, self).__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(input_size, output_size, bias=False)
        # self.fc2 = nn.Linear(hidden_size, output_size, bias=True)
        nn.init.xavier_normal_(self.fc1.weight)
        # nn.init.xavier_normal_(self.fc2.weight)
        # self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.fc1(x)
        return x


class DataFrameDataset(Dataset):
    def __init__(self, x, y):
        features = x
        targets = y
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def train(x_train, y_train, x_test, y_test):
    model = MLP(input_size=4096, output_size=1, hidden_size=512).cuda()
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.001)
    loss_fn = nn.MSELoss()
    train_dataset = DataFrameDataset(x_train, y_train)
    val_dataset = DataFrameDataset(x_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, drop_last=False)
    the_model = copy.deepcopy(model)
    max_pearsonr = 0.0
    for epoch in range(10000):
        for batch in train_loader:
            x, y = batch
            x = x.cuda()
            y = y.cuda()
            optimizer.zero_grad()
            output = model(x)
            loss = loss_fn(output.squeeze(), y)
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                val_pred = []
                val_true = []
                for batch in val_loader:
                    x, y = batch
                    x = x.cuda()
                    y = y.cuda()
                    output = model(x)
                    val_pred.extend(output.squeeze().tolist())
                    val_true.extend(y.tolist())
                    loss = loss_fn(output.squeeze(), y)
                    val_loss += loss.item()
            correlation = pearsonr(val_pred, val_true).correlation
            if correlation > max_pearsonr:
                max_pearsonr = correlation
                the_model = copy.deepcopy(model)
            model.train()
    return the_model


def predict(model, x_test):
    model.eval()
    dataset = DataFrameDataset(x_test, np.zeros(len(x_test)))
    test_loader = DataLoader(dataset, batch_size=512, shuffle=False, drop_last=False)

    all_preds = []
    for batch in test_loader:
        x, _ = batch
        x = x.cuda()
        with torch.no_grad():
            output = model(x)
            all_preds.append(output.cpu().numpy())

    return np.concatenate(all_preds).squeeze()


with open('processed_dataset_mixtral_8x7B_instruct_qlora_nf4_reflect_forward.pkl', 'rb') as f:
    data = pickle.load(f)

features = sorted(list(set(data[0].keys()) - {'target'}))
kf = KFold(n_splits=5, shuffle=True, random_state=6)

all_correlations = {}

for feature in features:
    if '17' not in feature:
        continue
    X = np.array([i[feature] for i in data]).squeeze()
    y = np.array([i['target'] for i in data]).squeeze()
    print(f"Running cross-validation for Layer {feature}")

    fold_corrs = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = train(X_train, y_train, X_test, y_test)
        y_pred = predict(model, X_test)
        corr = pearsonr(y_pred, y_test).correlation
        print(corr)
        fold_corrs.append(corr)

    avg_accuracy = np.mean(fold_corrs)
    all_correlations[feature] = avg_accuracy
    print(f"Layer {feature} - Average 5-fold Accuracy: {avg_accuracy}")

print(all_correlations)
