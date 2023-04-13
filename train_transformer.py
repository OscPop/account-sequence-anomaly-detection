import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder

import pandas as pd
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

# set pandas options

pd.set_option('display.float_format', lambda x: '%.2f' % x)

# Read data
df=pd.read_csv(Path('data/book.txt'), 
    sep='|', 
    header=0,
    encoding='utf-8', 
    low_memory=False, 
    decimal=',', 
    parse_dates=['voucher_date', 'last_update'])

df = df[df['voucher_type'] == 'VB']
df=df.sort_values(by=['voucher_no','sequence_no'])

# set account dtype to int
df['account'] = df['account'].astype(int)

# collapse by voucher_no and list account_no
df = df.groupby('voucher_no').agg(
    {'account':list}
)

# Replace with your actual data
sequences = df.account.tolist()

# Flatten the sequences and create a set of unique account numbers
flat_sequences = [account for seq in sequences for account in seq]
unique_accounts = set(flat_sequences)

# Encode the account numbers as integers
encoder = LabelEncoder()
encoder.fit(list(unique_accounts))

# save encoder
import pickle
with open('encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)


# Encode the sequences and pad them with -1 to a fixed length: maximum sequence length
max_seq_length = max([len(seq) for seq in sequences])
encoded_sequences = [encoder.transform(seq).tolist() for seq in sequences]
padded_sequences = [seq + [0] * (max_seq_length - len(seq)) for seq in encoded_sequences]
#padded_sequences = [seq + [0] * (max_seq_length - len(seq)) for seq in sequences]

# Create input-output pairs for training
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# Create input-output pairs for training
data = []
for seq in padded_sequences:
    for i in range(1, len(seq)):
        data.append((seq[:i], seq[i]))


class AccountSequenceDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
    targets_1d = [torch.tensor([t], dtype=torch.long) for t in targets]
    targets_padded = pad_sequence(targets_1d, batch_first=True, padding_value=0)
    return inputs_padded, targets_padded


dataset = AccountSequenceDataset(data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def generate_mask(self, size):
        mask = torch.tril(torch.ones(size, size) == 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x):
        embed = self.embedding(x)
        tgt_mask = self.generate_mask(x.size(1)).to(device)

        src_key_padding_mask = (x == 0)
        tgt_key_padding_mask = (x == 0)
        output = self.transformer(embed, embed, tgt_mask=tgt_mask, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        logits = self.fc(output)
        return logits


device = torch.device("cpu")

# Model and training parameters
vocab_size = len(unique_accounts) + 1
d_model = 128
nhead = 4
num_layers = 2
lr = 0.001
epochs = 10

model = TransformerModel(vocab_size, d_model, nhead, num_layers).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training loop
for epoch in range(epochs):

    model.train()
    for batch_idx, (x, y) in enumerate(dataloader):
        print('Batch index: ', batch_idx)
        print('Batch x size: ', x[0].size())
        print('Batch x label: ', x[1])
        print('Batch y size: ', y[0].size())
        print('Batch y label: ', y[1])

        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        logits_reshaped = logits.view(-1, logits.size(-1))
        y_reshaped = y.view(-1)
        valid_indices = y_reshaped != 0
        logits_masked = torch.masked_select(logits_reshaped, valid_indices.unsqueeze(1)).view(-1, logits.size(-1))
        y_masked = torch.masked_select(y_reshaped, valid_indices)
        loss = criterion(logits_masked, y_masked)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

# save the model
torch.save(model.state_dict(), 'models/LSTM_model.pt')

# apply outlier detection
def calculate_reconstruction_loss(model, dataloader, device):
    losses = []
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            losses.append(loss.item())
    return losses

reconstruction_losses = calculate_reconstruction_loss(model, dataloader, device)

import numpy as np

mean_loss = np.mean(reconstruction_losses)
std_loss = np.std(reconstruction_losses)
threshold = mean_loss + 2 * std_loss

outliers = [seq for seq, loss in zip(sequences, reconstruction_losses) if loss > threshold]

print(f"Number of outliers: {len(outliers)}")

# save outliers
with open('Result/outliers.pkl', 'wb') as f:
    pickle.dump(outliers, f)
