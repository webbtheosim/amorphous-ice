import numpy as np
from scipy.stats import multivariate_normal
from scipy.spatial.distance import cdist
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class Autoencoder(nn.Module):

    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 10 * input_dim),
            nn.Tanh(),
            nn.Linear(10 * input_dim, 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 10 * input_dim),
            nn.Tanh(),
            nn.Linear(10 * input_dim, input_dim)
        )

    def encode(self, X):
        return self.encoder(X)
    
    def decode(self, Z):
        return self.decoder(Z)
    
    def forward(self, X):
        Z = self.encoder(X)
        X_hat = self.decoder(Z)
        return X_hat

class AutoencoderGMM:

    def __init__(self, input_dim, latent_dim=2, num_classes=3):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.model = Autoencoder(self.input_dim, self.latent_dim)

    def _fit_autoencoder(self, X, max_epochs):

        train_loader, val_loader = self._prepare_loaders(X)

        criterion  = torch.nn.MSELoss()
        optimizer  = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)

        for epoch in range(max_epochs):

            train_loss = 0.0
            self.model.train()
            for batch_idx, (X,) in enumerate(train_loader):
                optimizer.zero_grad()
                X_hat = self.model(X)
                loss = criterion(X_hat, X)
                train_loss += loss.item()
                loss.backward()
                optimizer.step()

            val_loss = 0.0
            self.model.eval()
            with torch.no_grad():
                for batch_idx, (X,) in enumerate(val_loader):
                    X_hat = self.model(X)
                    loss = criterion(X_hat, X)
                    val_loss += loss.item()

            train_loss /= len(train_loader.dataset)
            val_loss /= len(val_loader.dataset)

            print(f'Epoch {epoch + 1} | Train loss = {train_loss:.8f} | Val loss =  {val_loss:.8f} | Rate = {optimizer.param_groups[0]["lr"]:.6f}')

            scheduler.step(val_loss)
            if optimizer.param_groups[0]['lr'] < 1e-5:
                print('Stopping early.')
                break

    def _prepare_loaders(self, X):

        # Prepare train/validation split.
        idx = [i for i in range(X.shape[0])]
        np.random.shuffle(idx)
        train_idx = idx[0:int(0.9 * X.shape[0])]
        val_idx = idx[int(0.9 * X.shape[0]):]
        X_train = X[train_idx]
        X_val = X[val_idx]

        # Scale based on training data.
        self.scaler = StandardScaler()
        self.scaler.fit(X_train)
        X_train_sc = self.scaler.transform(X_train)
        X_val_sc = self.scaler.transform(X_val)
        X_train_sc = torch.tensor(X_train_sc, dtype=torch.float32)
        X_val_sc = torch.tensor(X_val_sc, dtype=torch.float32)

        # Define datasets and dataloaders.
        train_dataset = TensorDataset(X_train_sc)
        val_dataset = TensorDataset(X_val_sc)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

        return train_loader, val_loader
    
    def _fit_gmm(self, Z, y):

        # Fit Gaussian probability distribution to HDA configurations.
        hda_idx = np.argwhere(y == 0).reshape(-1)
        Z_hda = Z[hda_idx]
        mean = np.mean(Z_hda, axis=0)
        cov = np.cov(Z_hda, rowvar=False)
        self.hda_prob = multivariate_normal(mean=mean, cov=cov)

        # Fit Gaussian probability distribution to LDA configurations.
        lda_idx = np.argwhere(y == 1).reshape(-1)
        Z_lda = Z[lda_idx]
        mean = np.mean(Z_lda, axis=0)
        cov = np.cov(Z_lda, rowvar=False)
        self.lda_prob = multivariate_normal(mean=mean, cov=cov)

        # Fit Gaussian probability distribution to liquid configurations.
        if self.num_classes == 3:
            liquid_idx = np.argwhere(y == 2).reshape(-1)
            Z_liquid = Z[liquid_idx]
            mean = np.mean(Z_liquid, axis=0)
            cov = np.cov(Z_liquid, rowvar=False)
            self.liquid_prob = multivariate_normal(mean=mean, cov=cov)

    def fit(self, X, y, max_epochs=100):

        # Fit autoencoder.
        X = torch.tensor(X, dtype=torch.float32)
        self._fit_autoencoder(X, max_epochs=max_epochs)

        # Fit GMM based on autoencoder embedding.
        X = self.scaler.transform(X)
        X = torch.tensor(X, dtype=torch.float32)
        Z = self.model.encode(X).cpu().detach().numpy()
        self._fit_gmm(Z, y)

    def predict(self, X, prob=False):

        # Get latent representations of inputs.
        X = self.scaler.transform(X)
        X = torch.tensor(X, dtype=torch.float32)
        Z = self.model.encode(X).cpu().detach().numpy()

        # Estimate probabilities using Monte Carlo integration.
        N_SAMPLES = 10000
        CUT = 0.1
        hda_samples = self.hda_prob.rvs(size=N_SAMPLES, random_state=1)
        lda_samples = self.lda_prob.rvs(size=N_SAMPLES, random_state=1)
        hda_dist = cdist(Z, hda_samples, metric='mahalanobis', VI=np.linalg.inv(self.hda_prob.cov))
        lda_dist = cdist(Z, lda_samples, metric='mahalanobis', VI=np.linalg.inv(self.lda_prob.cov))
        hda_probs = np.count_nonzero(np.where(hda_dist < CUT, 1, 0), axis=1) / N_SAMPLES
        lda_probs = np.count_nonzero(np.where(lda_dist < CUT, 1, 0), axis=1) / N_SAMPLES
        probs = np.hstack((hda_probs.reshape(-1,1), lda_probs.reshape(-1,1)))

        # Return result.
        if prob:
            return np.argmax(probs, axis=1), probs
        else:
            return np.argmax(probs, axis=1)
        
if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import KFold

    from utils import load_data
    from boo_nn import NeuralNetwork, train_neural_network, evaluate_neural_network
    from pointnet import PointNet, train_pointnet, evaluate_pointnet
    from autoencoder_gmm import AutoencoderGMM
    from probabilistic_model import ProbabilisticModel

    # Specify water model (either 'scan' or 'mbpol').
    MODEL = 'mbpol'

    # Save probabilites for outlier detection analysis.
    probabilities = {
        'boo-nn': {'hda': [], 'lda': []},
        'pointnet': {'hda': [], 'lda': []},
        'ae-gmm': {'hda': [], 'lda': []},
        'ours': {'hda': [], 'lda': []},
    }
    ice_probabilities = {
        'boo-nn': {'hda': [], 'lda': []},
        'pointnet': {'hda': [], 'lda': []},
        'ae-gmm': {'hda': [], 'lda': []},
        'ours': {'hda': [], 'lda': []},
    }

    np.random.seed(1)

    # Prepare datasets for AE-GMM.
    X, y = load_data(model='mbpol', feat='stein', states=['hda', 'lda', 'liquid'])
    hda_idx = np.argwhere(y == 0).reshape(-1)
    lda_idx = np.argwhere(y == 1).reshape(-1)
    liquid_idx = np.argwhere(y == 2).reshape(-1)
    X_liquid = X[liquid_idx]
    y_liquid = y[liquid_idx]
    all_idx = np.hstack((hda_idx, lda_idx))
    X = X[all_idx]
    y = y[all_idx]

    # Get a train/test split.
    metrics = {
        'hda_precisions': [],
        'lda_precisions': [],
        'hda_recalls': [],
        'lda_recalls': [],
        'accuracies': []
    }
    prob_lda = []
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    for idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f'Evaluating fold {idx + 1} / 5.')
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]
        X_train = np.vstack((X_train, X_liquid))
        y_train = np.hstack((y_train, y_liquid))
        model = AutoencoderGMM(input_dim=X.shape[1], latent_dim=2, num_classes=3)
        model.fit(X_train, y_train, max_epochs=10)
        y_pred, probs = model.predict(X_test, prob=True)   
        probs = np.log(probs + 1e-100) 

        # Save probabilities for true configurations for subsequent analysis.
        lda_idx = np.argwhere(y_test == 1).reshape(-1) 
        hda_idx = np.argwhere(y_test == 0).reshape(-1) 
        lda_probs = probs[lda_idx, 1]
        hda_probs = probs[hda_idx, 0]
        probabilities['ae-gmm']['lda'].append(lda_probs)
        probabilities['ae-gmm']['hda'].append(hda_probs)      

        y_pred = np.argmax(probs[:,0:2], axis=1) # Only consider HDA/LDA predictions.

        # Save performance metrics.
        conf_mat = confusion_matrix(y_test, y_pred)
        metrics['hda_precisions'].append(conf_mat[0,0] / (conf_mat[1,0] + conf_mat[0,0]))
        metrics['lda_precisions'].append(conf_mat[1,1] / (conf_mat[0,1] + conf_mat[1,1]))
        metrics['hda_recalls'].append(conf_mat[0,0] / (conf_mat[0,1] + conf_mat[0,0]))
        metrics['lda_recalls'].append(conf_mat[1,1] / (conf_mat[1,0] + conf_mat[1,1]))
        metrics['accuracies'].append((conf_mat[0,0] + conf_mat[1,1]) / np.sum(conf_mat, axis=(0,1)))