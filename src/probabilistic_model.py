import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KernelDensity
import torch

class ProbabilisticModel:
    '''
        A differentiable implementation of the naive Bayes classifier applied to kernel
        density estimations of class distributions along representative but independent
        descriptors.
    '''

    def __init__(self, max_features=10, corr_cut=0.8, include=0.98, detect_outliers=True, use_features=None, ignore_feat=0):
        self.max_features = max_features
        self.include = include
        self.detect_outliers = detect_outliers
        self.corr_cut = corr_cut

        # If desired, manually set which features are used.
        self.use_features = use_features
        self.ignore_feat = ignore_feat

    class KernelDensityEstimate:
        '''
            A continuous and differentiable distribution fit to a provided set of
            samples. Provides an evaluation of the kernel density estimate normalized
            so that the maximum value is 1.0. 
        '''

        def __init__(self, v):
            '''
                Estimate the bandwidth based on log likelihood maximization applied to a
                subsample of the provided data.
            '''
            self.v = v.reshape(-1,1)
            v_subsample = np.random.choice(v.detach().cpu().numpy(), size=500, replace=False)
            params = {'bandwidth': np.logspace(-3, 1, 30)}
            grid = GridSearchCV(KernelDensity(), params)
            grid.fit(v_subsample.reshape(-1,1))
            self.bandwidth = grid.best_params_['bandwidth']

        def evaluate(self, V, dx_factor=0.5):
            '''Evaluate the kernel density estimate at the provided inputs.'''
            V = V.reshape(-1,1,1)
            V_hi = V + dx_factor * self.bandwidth 
            Z_hi = (V_hi - self.v.unsqueeze(0)) / self.bandwidth
            cum_hi = 0.5 * (1.0 + torch.erf(Z_hi / torch.sqrt(torch.tensor([2.0]))))
            V_lo = V - dx_factor * self.bandwidth
            Z_lo = (V_lo - self.v.unsqueeze(0)) / self.bandwidth
            cum_lo = 0.5 * (1.0 + torch.erf(Z_lo / torch.sqrt(torch.tensor([2.0]))))
            integral = cum_hi - cum_lo
            prob = torch.sum(integral, dim=1) / self.v.shape[0]
            return prob.reshape(-1)

    def fit(self, X, y):

        # Perform train/validation splitting.
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=1)

        # Check if user has specified features.
        if self.use_features is not None:

            # Use said features.
            keep_idx = self.use_features

            # Compute mutual information for said features.
            mut_inf = []
            n_bins = int(X_train.shape[0]**(1./3.))
            for idx in range(X_train.shape[1]):
                if idx in keep_idx:
                    feat = X_train[:,idx]
                    try:
                        feat_binned = np.digitize(feat, bins=np.histogram_bin_edges(feat, bins=n_bins)) - 1
                        joint_counts = np.histogram2d(feat_binned, y_train, bins=[np.arange(n_bins + 1), np.arange(np.max(y_train) + 2)])[0]
                        p_xy = joint_counts / np.sum(joint_counts)
                        p_x = np.sum(p_xy, axis=1, keepdims=True)
                        p_y = np.sum(p_xy, axis=0, keepdims=True)
                        nz = p_xy > 0
                        mi = np.sum(p_xy[nz] * np.log(p_xy[nz] / (p_x @ p_y)[nz]))
                        mut_inf.append(mi)
                    except:
                        mut_inf.append(0.0)
                else:
                    mut_inf.append(0.0)
            mut_inf = np.array(mut_inf)
            self.mut_inf = mut_inf

        # Perform feature selection.
        else:    

            # Compute mutual information for every feature.
            mut_inf = []
            n_bins = int(X_train.shape[0]**(1./3.))
            for idx in range(X_train.shape[1]):
                feat = X_train[:,idx]
                try:
                    feat_binned = np.digitize(feat, bins=np.histogram_bin_edges(feat, bins=n_bins)) - 1
                    joint_counts = np.histogram2d(feat_binned, y_train, bins=[np.arange(n_bins + 1), np.arange(np.max(y_train) + 2)])[0]
                    p_xy = joint_counts / np.sum(joint_counts)
                    p_x = np.sum(p_xy, axis=1, keepdims=True)
                    p_y = np.sum(p_xy, axis=0, keepdims=True)
                    nz = p_xy > 0
                    mi = np.sum(p_xy[nz] * np.log(p_xy[nz] / (p_x @ p_y)[nz]))
                    mut_inf.append(mi)
                except:
                    mut_inf.append(0.0)
            mut_inf = np.array(mut_inf)

            # Compute correlations among features.
            similarities = np.zeros((X_train.shape[1], X_train.shape[1]))
            for feat1 in range(X_train.shape[1]):
                similarities[feat1, feat1] = 1.0
                for feat2 in range(feat1 + 1, X_train.shape[1]):
                    corr = np.abs(pearsonr(X_train[:,feat1], X_train[:,feat2]).statistic)
                    similarities[feat1, feat2] = corr
                    similarities[feat2, feat1] = corr

            # Sort independent indices by mutual information.
            feat_idx = np.array([i for i in range(X.shape[1])], dtype=np.int32)
            sorted_ids = np.argsort(-mut_inf).reshape(-1)
            feat_idx = feat_idx[sorted_ids]
            mut_inf_sorted = mut_inf[sorted_ids]

            # Choose features to keep.
            mi_cut = 0.1 * mut_inf[0]
            candidate_idx = [idx for idx, mi in zip(feat_idx, mut_inf_sorted) if mi > mi_cut]
            keep_idx = []
            for i in candidate_idx:
                if len(keep_idx) > 0:
                    max_corr = np.max(similarities[i, keep_idx])
                    if max_corr < self.corr_cut:
                        keep_idx.append(i)
                else:
                    keep_idx.append(i)
            keep_idx = keep_idx[0:self.max_features]

            # Ignore best features if specified; used for ablation analyses.
            if self.ignore_feat > 0:
                keep_idx = keep_idx[self.ignore_feat:]

            # # Choose those indices based on a cutoff.
            # cutoff = 0.1 * mut_inf[0]
            # keep_idx = [idx for idx, mi in zip(feat_idx, mut_inf) if mi > cutoff]
            # if len(keep_idx) > self.max_features:
            #     keep_idx = keep_idx[0:self.max_features]

        # Get probability distributions for each chosen feature for each class, 
        # implemented in PyTorch.
        self.classes = np.unique(y_train).astype(np.int8)
        eval_funcs = {c: [] for c in self.classes}
        for idx in keep_idx:
            for c in self.classes:
                class_idx = np.argwhere(y_train == c).reshape(-1)
                v = torch.tensor(X_train[class_idx,idx], dtype=torch.float)
                kde = self.KernelDensityEstimate(v=v)
                eval_funcs[c].append(kde)

        # Save values for future use.
        self.keep_idx = keep_idx
        self.mutual_information = mut_inf[keep_idx]
        self.eval_funcs = eval_funcs
        self.chosen_features = keep_idx

        # Evaluate log probabilities on validation set.
        X_val = torch.tensor(X_val, dtype=torch.float)
        val_prob = torch.ones(size=(X_val.shape[0], len(self.keep_idx), len(self.classes)))
        for i, idx in enumerate(self.keep_idx):
            for c in self.classes:
                val_prob[:,i,c] = self.eval_funcs[c][i].evaluate(X_val[:,idx].reshape(-1,1))
            
        # Determine cutoffs for each class.
        self.cutoffs = {}
        val_prob = torch.sum(torch.log(val_prob + 1e-30), dim=1)
        for c in self.classes:
            class_idx = np.argwhere(y_val == c).reshape(-1)
            self.cutoffs[c] = np.percentile(val_prob[class_idx,c], q=100 * (1.0 - self.include))

    def estimate(self, X):
        '''Compute class probabilities for each entry in X.'''
        X = torch.tensor(X, dtype=torch.float)
        prob = torch.ones(size=(X.shape[0], len(self.keep_idx), len(self.classes)))
        for i, idx in enumerate(self.keep_idx):
            for c in self.classes:
                prob[:,i,c] = self.eval_funcs[c][i].evaluate(X[:,idx])
        return prob
    
    def get_log_prob(self, X, binary=False):
        '''Get labels for each entry in X.'''
        prob = self.estimate(X)
        class_log_prob = torch.sum(torch.log(prob + 1e-30), dim=1).numpy()
        if binary and class_log_prob.shape[1] > 2:
            class_log_prob = class_log_prob[:,0:2]
        return class_log_prob
        
    def predict(self, X, binary=False):
        '''Get labels for each entry in X.'''

        # Get probabilities of belonging to each class.
        prob = self.estimate(X)
        class_log_prob = torch.sum(torch.log(prob + 1e-30), dim=1).numpy()

        # Handle outlier detection.
        if self.detect_outliers:
            for c in self.classes:
                outlier_idx = np.argwhere(class_log_prob[:,c] < self.cutoffs[c]).reshape(-1)
                class_log_prob[outlier_idx,c] = -np.inf

        # If binary is indicated, we only want HDA/LDA predictions. Otherwise, you can
        # get predictions for every class in the training set.
        if binary and class_log_prob.shape[1] > 2:
            class_log_prob = class_log_prob[:,0:2]

        # Get labels and modify for outliers.
        labels = np.argmax(class_log_prob, axis=1)
        if self.detect_outliers:
            mask = np.all(class_log_prob == -np.inf, axis=1)
            idx = np.where(mask)[0]
            labels[idx] = -1

        return labels
    
    def confidence(self, X, binary=False):
        '''Get log(scores) for each entry in X.'''

        # Get probabilities of belonging to each class.
        prob = self.estimate(X)
        class_log_prob = torch.sum(torch.log(prob + 1e-100), dim=1).numpy()

        # Use cutoffs to determine outliers.
        if self.detect_outliers:
            for c in self.classes:
                outlier_idx = np.argwhere(class_log_prob[:,c] < self.cutoffs[c]).reshape(-1)
                class_log_prob[outlier_idx,c] = -np.inf

        # If binary is indicated, we only want HDA/LDA predictions. Otherwise, you can
        # get predictions for every class in the training set.
        if binary and class_log_prob.shape[1] > 2:
            class_log_prob = class_log_prob[:,0:2]

        return np.exp(class_log_prob)