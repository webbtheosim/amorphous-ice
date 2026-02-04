import argparse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim

class NeuralNetwork(nn.Module):
    '''
        A neural network implementation similar to the one used by 
        Martelli et al.
    '''

    def __init__(self, n_classes=3):
        super().__init__()
        self.fc1 = nn.Linear(30,10)
        self.act1 = nn.Sigmoid()
        self.fc2 = nn.Linear(10,n_classes)
        self.act2 = nn.Sigmoid()
        self.init_weights()
        self.scaler = None

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, X):
        return self.act2(self.fc2(self.act1(self.fc1(X))))
    
def train_neural_network(model, X, y, print_freq=-1):
    '''
        Method for training a neural network consistent with the implementation
        used by Martelli et al.
    '''

    # Get a train/val split.
    idx = [i for i in range(X.shape[0])]
    np.random.shuffle(idx)
    cutoff = int(0.9 * X.shape[0])
    train_idx = idx[:cutoff]
    val_idx = idx[cutoff:]
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_val = X[val_idx]
    y_val = y[val_idx]

    # Scale training data.
    sc = StandardScaler().fit(X_train)
    X_train = sc.transform(X_train)
    X_val = sc.transform(X_val)
    model.scaler = sc

    # Convert data to torch tensors.
    X_train = torch.tensor(X_train, dtype=torch.float)
    y_train = torch.tensor(y_train, dtype=torch.float)
    X_val = torch.tensor(X_val, dtype=torch.float)
    y_val = torch.tensor(y_val, dtype=torch.float)

    # Define training parameters.
    criterion = nn.MSELoss()
    optimizer = optim.Adam(params=model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=20)
    batch_size = 1000

    # Train model using manual batch creation.
    train_idx = [i for i in range(X_train.shape[0])]
    val_idx = [i for i in range(X_val.shape[0])]
    for epoch in range(10000):

        np.random.shuffle(train_idx)
        np.random.shuffle(val_idx)
        n_train_batches = int(len(train_idx) / batch_size) if len(train_idx) % batch_size == 0 else int(len(train_idx) / batch_size) + 1
        n_val_batches = int(len(val_idx) / batch_size) if len(val_idx) % batch_size == 0 else int(len(val_idx) / batch_size) + 1

        # Fit to training data.
        train_loss = 0.0
        model.train()
        for batch_num in range(n_train_batches):

            start = batch_num * batch_size
            stop = min((batch_num + 1) * batch_size, len(train_idx))
            batch_idx = train_idx[start:stop]

            optimizer.zero_grad()
            y_hat = model(X_train[batch_idx])
            loss = criterion(y_train[batch_idx], y_hat)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        # Evaluate on validation data.
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch_num in range(n_val_batches):

                start = batch_num * batch_size
                stop = min((batch_num + 1) * batch_size, len(train_idx))
                batch_idx = val_idx[start:stop]

                y_hat = model(X_val[batch_idx])
                loss = criterion(y_val[batch_idx], y_hat)
                val_loss += loss.item()

        train_loss /= X_train.shape[0]
        val_loss /= X_val.shape[0]

        # Report progress.
        if print_freq != -1:
            print(f'Epoch {epoch + 1} | Train loss = {train_loss:.8f} | Val loss =  {val_loss:.8f} | Rate = {optimizer.param_groups[0]["lr"]:.6f}')

        # Stop training early if learning rate decreases below cutoff.
        scheduler.step(val_loss)
        if optimizer.param_groups[0]['lr'] < 1e-5:
            if print_freq != -1:
                print('Stopping early.')
            break

def evaluate_neural_network(model, X, prob=False):
    '''
        Method for getting predicted labels from neural network.
    '''

    X_sc = model.scaler.transform(X)
    X_sc = torch.tensor(X_sc, dtype=torch.float)
    probs = model(X_sc).detach().cpu().numpy()

    if not prob:
        return np.argmax(probs, axis=1)
    else:
        return np.argmax(probs, axis=1), probs