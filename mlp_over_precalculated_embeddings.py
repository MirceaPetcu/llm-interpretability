import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import random_split
from scipy.stats import pearsonr
import numpy as np


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(MLP, self).__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(input_size, output_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, output_size, bias=True)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.fc1(x)
        # x = self.relu(x)
        # x = self.dropout(x)
        # x = self.fc2(x)
        return x


def get_data():
    import pickle
    with open('processed_dataset_mixtral_8x7B_instruct_qlora_nf4_reflect_forward.pkl', 'rb') as f:
        data = pickle.load(f)
    return data


import torch
from torch.utils.data import Dataset, DataLoader

# embeddgins_17
class DataFrameDataset(Dataset):
    def __init__(self, dataframe):
        features = np.array([[v for k, v in i.items() if k == 'embeddgins_17'] for i in dataframe]).squeeze()
        targets = np.array([[v for k, v in i.items() if k == 'target'] for i in dataframe]).squeeze()
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


if __name__ == '__main__':
    df = get_data()
    model = MLP(input_size=4096, output_size=1, hidden_size=512).cuda()
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.001)
    loss_fn = nn.MSELoss()
    dataset = DataFrameDataset(df)
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size

    generator1 = torch.Generator().manual_seed(40)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator1)

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, drop_last=False)

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
                    print(f'Epoch {epoch}/{100}, Loss: {val_loss/len(val_loader):.4f}')
                correlation = pearsonr(val_pred, val_true).correlation
                if correlation > max_pearsonr:
                    max_pearsonr = correlation
                    torch.save(model.state_dict(), 'mlp.pth')
                print(f'Correlation: {correlation:.4f}')
                print(f'pvalue: {pearsonr(val_pred, val_true).pvalue:.4f}')
                model.train()

    print(f'Max Pearson Correlation: {max_pearsonr:.4f}')




