import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import torch.nn as nn
from scipy.stats import pearsonr
import torch
from torch.utils.data import Dataset, DataLoader
import copy
from utils import prepare_input, get_word_by_index
import argparse
from transformers import AutoTokenizer


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(MLP, self).__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(input_size, output_size, bias=False)
        nn.init.xavier_normal_(self.fc1.weight)

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


def parse_args():
    parser = argparse.ArgumentParser(description='Run cross-validation on the extracted features')
    parser.add_argument('--data', type=str,
                        default='processed_dataset_mixtral_8x7B_instruct_qlora_nf4_forward_with_tokens.pkl',
                        help='Path to the processed data file')
    parser.add_argument('--model', type=str, default='mistralai/Mixtral-8x7B-Instruct-v0.1',
                        help='Model ID')
    parser.add_argument('--word', type=str, default='Reflect',
                        help='Word of interest')
    parser.add_argument('--nth_word', type=int, default=3,
                        help='Index of the word of interest')
    return parser.parse_args()



if __name__ == '__main__':
    args = parse_args()
    with open(args.data, 'rb') as f:
        data = pickle.load(f)
    words_of_interest = get_word_by_index(args.nth_word)
    data = prepare_input(data, args.model, words_of_interest=words_of_interest)

    features = sorted(list(set(data[0].keys()) - {'target', 'tokens'}))
    kf = KFold(n_splits=5, shuffle=True, random_state=6)

    all_correlations = {}

    for feature in features:
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
