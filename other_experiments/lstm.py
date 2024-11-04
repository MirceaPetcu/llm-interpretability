import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pickle
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import pearsonr
from tqdm import tqdm


# Sample data generation
num_samples = 1000  # Total number of samples
with open('../processed_dataset_llama3_8b.pkl', 'rb') as f:
    data = pickle.load(f)
X = [np.array([v for k, v in i.items() if k != 'target']).squeeze() for i in data]
y = [i['target'] for i in data]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Create TensorDataset and DataLoader
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # Use batch_size and shuffle for SGD
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


class ShallowLSTM(nn.Module):
    def __init__(self, input_size=4096, hidden_size=512, num_layers=2):
        super(ShallowLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # Output layer for regression

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Use the last time step output for regression
        return out


# Create the model
model = ShallowLSTM().to('cuda')

# Define loss function and optimizer
criterion = nn.MSELoss().to('cuda')
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
for epoch in tqdm(range(num_epochs)):
    model.train()
    running_loss = 0.0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to('cuda'), batch_y.to('cuda')

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs.squeeze(), batch_y)  # Squeeze to match dimensions
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.inference_mode():
            y_pred = []
            y_true = []
            for x_val, y_val in test_loader:
                x_val, y_val = x_val.to('cuda'), y_val.to('cuda')
                y_hat = model(x_val)
                test_loss = criterion(y_hat.squeeze(), y_val)
                y_pred.extend(y_hat.cpu().detach().numpy().tolist())
                y_true.extend(y_val.cpu().detach().numpy().tolist())
            print(pearsonr(y_true, y_pred).correlation)
            print(pearsonr(y_true, y_pred).pvalue)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')
        model.train()
