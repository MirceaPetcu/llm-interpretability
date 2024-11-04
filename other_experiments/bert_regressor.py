import pandas as pd
import torch
from transformers import BertTokenizerFast, BertModel, BertForSequenceClassification
from transformers import AdamW, get_scheduler
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import datasets
from utils import get_pearson_correlation
import numpy as np
from scipy.stats import pearsonr


# Load the dataset
df = pd.read_csv('../bold_response_LH.csv')

# Ensure the target (last column) is numeric for regression
df['target'] = pd.to_numeric(df.iloc[:, -1])

# Split the dataset into train and test
train_df, test_df = train_test_split(df, test_size=0.2)

# Convert pandas dataframe to Hugging Face dataset
train_dataset = datasets.Dataset.from_pandas(train_df)
test_dataset = datasets.Dataset.from_pandas(test_df)

# Load the model and tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1).cuda()

# Tokenize the text data
def tokenize_function(examples):
    return tokenizer(examples['sentence'], padding="max_length", truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Convert the datasets to PyTorch tensors
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'target'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'target'])

# Create DataLoader for PyTorch
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=32)
test_dataloader = DataLoader(test_dataset, batch_size=32)

from transformers import BertForSequenceClassification

# Load BERT with a regression head (num_labels=1)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1).cuda()


from torch.optim import AdamW
from transformers import get_scheduler
import torch.nn as nn

for name, param in model.base_model.named_parameters():
    print(name)
    param.requires_grad = False

# Use AdamW optimizer
optimizer = AdamW(model.parameters(), lr=4e-5)

# Define a scheduler for learning rate decay
num_epochs = 10
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=100,
    num_training_steps=num_training_steps,
)

# Mean Squared Error Loss for regression
loss_fn = nn.MSELoss()

from tqdm.auto import tqdm

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['target'])
        loss = outputs.loss

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        lr_scheduler.step()
        progress_bar.update(1)

    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss / len(train_dataloader):.4f}")


    # Evaluation (Optional)
    model.eval()
    total_loss = 0
    y_pred = []
    y_true = []
    for batch in test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            predictions = outputs.logits.squeeze()
            loss = loss_fn(predictions, batch['target'])
            y_pred.extend(predictions.cpu().tolist())
            total_loss += loss.item()
            y_true.extend(batch['target'].cpu().tolist())

    avg_test_loss = total_loss / len(test_dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Test Loss: {avg_test_loss:.4f}")
    result = pearsonr(y_true, y_pred)

    print(f"Pearson Correlation: {result.correlation:.4f}")
    print('P-value:', result.pvalue.item())
