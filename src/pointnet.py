from e3nn.o3 import rand_matrix
import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import TensorDataset, DataLoader

def atom_types_to_ohe(data, types=[1,2]):
    '''Featurize environments using coordinates and OHE atom types.'''
    new_data = np.zeros((data.shape[0], data.shape[1], 3 + len(types)))
    new_data[:,:,0:3] = data[:,:,1:]
    for env_idx in range(new_data.shape[0]):
        for atom_idx in range(new_data.shape[1]):
            for type_idx, type in enumerate(types):
                if data[env_idx, atom_idx, 0] == type:
                    new_data[env_idx, atom_idx, type_idx + 3] = 1
    return new_data

def scale_distances(data):
    '''Scale distances in atomic environment so max. distance is 1.'''
    new_data = np.zeros((data.shape[0], data.shape[1], data.shape[2]))
    for env_idx in range(new_data.shape[0]):
        factor = 1.0 / np.linalg.norm(data[env_idx, 1, 0:3])
        new_data[env_idx,:,3:] = data[env_idx,:,3:]
        new_data[env_idx,:,0:3] = factor * data[env_idx,:,0:3]
    return new_data

def rotate_batch(pc):
    '''Randomly rotate a batch of environments during training.'''
    # input is B x n x 3
    batch_size = pc.size()[0]
    rot_mat = rand_matrix(batch_size)
    # (Bx3x3) x (Bx3xn) -> (Bx3xn) -> (Bxnx3)
    rot_pc = torch.transpose(torch.bmm(rot_mat, torch.transpose(pc[:, :, :3], 1, 2)), 1, 2)
    # re-append any extra atom ID information and transpose to (Bx3xn)
    rotated = torch.transpose(torch.concatenate((rot_pc, pc[:, :, 3:]), axis=2), 1, 2)
    return rotated

class PointNet(nn.Module):

    def __init__(self, in_dim, num_points, num_classes, conv2d):

        super(PointNet, self).__init__()
        self.in_dim = in_dim
        self.npts = num_points
        self.num_classes = num_classes
        self.conv2d = conv2d

        if self.conv2d:
            self.inp = nn.Sequential(
                nn.Conv2d(in_dim, 64, (3, 1), padding=(1,0)),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.Conv2d(64, 64, 1, padding=0),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.Conv2d(64, 64, 1, padding=0),
                nn.BatchNorm2d(64),
                nn.ReLU(True)
                )
            self.conv2 = nn.Sequential(
                nn.Conv2d(64, 128, 1, padding=0),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.Conv2d(128, 1024, 1, padding=0),
                nn.BatchNorm2d(1024),
                nn.ReLU(True)
                )
            self.invar_pool = nn.MaxPool2d((num_points, 1), stride=(2,2))

        else:
            self.conv1 = nn.Conv1d(in_dim, 64, 1)
            self.conv2 = nn.Conv1d(64, 64, 1)
            self.conv3 = nn.Conv1d(64, 64, 1)
            self.conv4 = nn.Conv1d(64, 128, 1)
            self.conv5 = nn.Conv1d(128, 1024, 1)

            self.bn1 = nn.BatchNorm1d(64)
            self.bn2 = nn.BatchNorm1d(64)
            self.bn3 = nn.BatchNorm1d(64)
            self.bn4 = nn.BatchNorm1d(128)
            self.bn5 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.bfc1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bfc2 = nn.BatchNorm1d(256)
        self.drp = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x is Bx3xP
        assert(x.size()[-1] == self.npts), 'Invalid number of point descriptors'

        if self.conv2d:
            x = x.unsqueeze_(-1) # Bx3xPx1
            x = self.inp(x)
            x = self.conv2(x)
            x = self.invar_pool(x)
            x = x.view(-1, 1024)
            x = self.relu(self.bfc1(self.fc1(x)))
            x = self.relu(self.bfc2(self.fc2(x)))
            #if self.training:
            #    x = self.drp(x)
            return self.fc3(x) # B x NC

        else:
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.relu(self.bn3(self.conv3(x)))
            x = self.relu(self.bn4(self.conv4(x)))
            x = self.bn5(self.conv5(x))
            x = torch.max(x, 2, keepdim=True)[0]
            x = x.view(-1, 1024)
            x = self.relu(self.bfc1(self.fc1(x)))
            x = self.relu(self.bfc2(self.fc2(x)))
            #if self.training:
            #    x = self.drp(x)
            return self.fc3(x) # B x NC
        
def train_pointnet(model, X, y, num_epochs=10, device='cpu'):
    '''Fit a PointNet model to inputs X and labels y.'''

    # Prepare training and validation sets.
    X = atom_types_to_ohe(X)
    X = scale_distances(X)
    idx = [i for i in range(X.shape[0])]
    np.random.shuffle(idx)
    train_idx = idx[0:int(0.9 * X.shape[0])]
    val_idx = idx[int(0.9 * X.shape[0]):]
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_val = X[val_idx]
    y_val = y[val_idx]

    # Convert to tensors.
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)

    # Prepare datasets and dataloaders.
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    # Define training parameters.
    optimizer  = optim.Adam(model.parameters(), lr=1e-3)
    criterion  = nn.BCEWithLogitsLoss()
    max_epochs = num_epochs

    # Fit model.
    for epoch in range(max_epochs):

        model.train()
        train_loss = 0.0
        for batch_idx, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            X = rotate_batch(X)
            y_pred = model(X)
            loss = criterion(y_pred, y)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (X, y) in enumerate(val_loader):
                X, y = X.to(device), y.to(device)
                X = rotate_batch(X)
                y_pred = model(X)
                loss = criterion(y_pred, y)
                val_loss += loss.item()

        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)

        print(f'Epoch {epoch + 1} | Train loss = {train_loss:.8f} | Val loss =  {val_loss:.8f} | Rate = {optimizer.param_groups[0]["lr"]:.6f}')

def evaluate_pointnet(model, X, prob=False, device='cpu'):
    '''Evaluate a fitted PointNet model on X.'''
    X = atom_types_to_ohe(X)
    X = scale_distances(X)
    X = torch.tensor(X, dtype=torch.float32, device=device)
    X = rotate_batch(X)
    probs = model(X).detach().cpu().numpy()
    if not prob:
        return np.argmax(probs, axis=1)
    else:
        return np.argmax(probs, axis=1), probs
    
if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import KFold
    np.random.seed(1)

    # Identify device for training.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Model training and evaluation performed on: {device}')

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
    
    # Prepare datasets for PointNet.
    NEIGH_SIZE = 16  # Default neighborhood size
    X_hda = np.load(f'../data/descriptors/neigh_{NEIGH_SIZE}/{MODEL}_hda_coords.npy')
    X_lda = np.load(f'../data/descriptors/neigh_{NEIGH_SIZE}/{MODEL}_lda_coords.npy')
    X_liquid = np.load(f'../data/descriptors/neigh_{NEIGH_SIZE}/{MODEL}_liquid_coords.npy')
    y_hda = np.zeros((X_hda.shape[0], 3))
    y_lda = np.zeros((X_lda.shape[0], 3))
    y_liquid = np.zeros((X_liquid.shape[0], 3))
    y_hda[:,0]    = 1
    y_lda[:,1]    = 1
    y_liquid[:,2] = 1

    # # TESTING.
    # NUM = 100
    # hda_idx = [i for i in range(X_hda.shape[0])]
    # np.random.shuffle(hda_idx)
    # X_hda = X_hda[hda_idx[0:NUM]]
    # y_hda = y_hda[hda_idx[0:NUM]]
    # lda_idx = [i for i in range(X_lda.shape[0])]
    # np.random.shuffle(lda_idx)
    # X_lda = X_lda[lda_idx[0:NUM]]
    # y_lda = y_lda[lda_idx[0:NUM]]
    # liquid_idx = [i for i in range(X_liquid.shape[0])]
    # np.random.shuffle(liquid_idx)
    # X_liquid = X_liquid[liquid_idx[0:NUM]]
    # y_liquid = y_liquid[liquid_idx[0:NUM]]

    # Finish preparing datasets for PointNet.
    X = np.vstack((X_hda, X_lda))
    y = np.vstack((y_hda, y_lda))

    # Assess classification accuracy for PointNet.
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
        y_train = np.vstack((y_train, y_liquid))

        model = PointNet(in_dim=5, num_points=51, num_classes=3, conv2d=True)
        model.to(device)

        train_pointnet(model, X_train, y_train, num_epochs=20, device=device)
        _, probs = evaluate_pointnet(model, X_test, prob=True, device=device)

        # Implement softmax on logits for probabilities; only consider HDA and LDA predictions.
        probs = probs[:,0:2]
        probs = 1. / (1. + np.exp(-probs))

        # Save probabilities for true configurations for subsequent analysis.
        lda_idx = np.argwhere(np.argmax(y_test, axis=1) == 1).reshape(-1) 
        hda_idx = np.argwhere(np.argmax(y_test, axis=1) == 0).reshape(-1) 
        lda_probs = probs[lda_idx, 1]
        hda_probs = probs[hda_idx, 0]
        probabilities['pointnet']['lda'].append(lda_probs)
        probabilities['pointnet']['hda'].append(hda_probs)
        y_pred = np.argmax(probs, axis=1) # Only consider HDA/LDA predictions.
        y_true = np.argmax(y_test, axis=1)
        
        # Save performance metrics.
        conf_mat = confusion_matrix(y_true, y_pred)
        metrics['hda_precisions'].append(conf_mat[0,0] / (conf_mat[1,0] + conf_mat[0,0]))
        metrics['lda_precisions'].append(conf_mat[1,1] / (conf_mat[0,1] + conf_mat[1,1]))
        metrics['hda_recalls'].append(conf_mat[0,0] / (conf_mat[0,1] + conf_mat[0,0]))
        metrics['lda_recalls'].append(conf_mat[1,1] / (conf_mat[1,0] + conf_mat[1,1]))
        metrics['accuracies'].append((conf_mat[0,0] + conf_mat[1,1]) / np.sum(conf_mat, axis=(0,1)))

    # Display accuracy metrics.
    print(f'HDA Precision: {np.mean(metrics["hda_precisions"]):.3f} +/- {np.std(metrics["hda_precisions"]):.3f}')
    print(f'LDA Precision: {np.mean(metrics["lda_precisions"]):.3f} +/- {np.std(metrics["lda_precisions"]):.3f}')
    print(f'HDA Recall: {np.mean(metrics["hda_recalls"]):.3f} +/- {np.std(metrics["hda_recalls"]):.3f}')
    print(f'LDA Recall: {np.mean(metrics["lda_recalls"]):.3f} +/- {np.std(metrics["lda_recalls"]):.3f}')
    print(f'Accuracy: {np.mean(metrics["accuracies"]):.3f} +/- {np.std(metrics["accuracies"]):.3f}')

    # Save probabilities.
    print(np.hstack(probabilities['pointnet']['hda']))
    print(np.hstack(probabilities['pointnet']['lda']))
    np.save('../notebooks/data/pointnet_hda_prob.npy', np.hstack(probabilities['pointnet']['hda']))
    np.save('../notebooks/data/pointnet_lda_prob.npy', np.hstack(probabilities['pointnet']['lda']))

    # Retrain PointNet model on all HDA, LDA, and liquid configurations.
    X = np.vstack((X, X_liquid))
    y = np.vstack((y, y_liquid))
    model = PointNet(in_dim=5, num_points=51, num_classes=3, conv2d=True)
    train_pointnet(model, X_train, y_train, num_epochs=100)

    # Evaluate trained PointNet when extrapolating to hexagonal ice structures.
    X_ice = np.load(f'../data/descriptors/neigh_{NEIGH_SIZE}/{MODEL}_ice_coords.npy')
    y_ice, prob_ice = evaluate_pointnet(model, X_ice, prob=True)
    prob_ice = prob_ice[:,0:2]
    prob_ice = 1. / (1. + np.exp(-prob_ice))
    ice_probabilities['pointnet']['hda'] = prob_ice[:,0]
    ice_probabilities['pointnet']['lda'] = prob_ice[:,1]

    # Save ice probabilities.
    print(ice_probabilities['pointnet']['hda'])
    print(ice_probabilities['pointnet']['lda'])
    np.save('../notebooks/data/pointnet_ice_hda_prob.npy', ice_probabilities['pointnet']['hda'])
    np.save('../notebooks/data/pointnet_ice_lda_prob.npy', ice_probabilities['pointnet']['lda'])