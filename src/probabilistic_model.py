import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KernelDensity
import torch

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
        rng = np.random.default_rng(seed=1)
        v_subsample = rng.choice(v.detach().cpu().numpy(), size=500, replace=False)
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

class ProbabilisticModel:
    '''
        A differentiable implementation of the naive Bayes classifier applied to kernel
        density estimations of class distributions along representative but independent
        descriptors.
    '''

    def __init__(
            self,
            n_feat=10,
            n_outlier=10,
            class_feat=None,
            outlier_feat=None,
            corr_cut=0.8,
            ignore_feat=0,
        ):
        self.n_feat = n_feat
        self.n_outlier = n_outlier
        self.class_feat = class_feat
        self.outlier_feat = outlier_feat
        self.corr_cut = corr_cut
        self.ignore_feat = ignore_feat

    def fit(self, X, y):

        # Compute mutual information for every feature.
        mut_inf = []
        n_bins = int(X.shape[0]**(1./3.))
        for idx in range(X.shape[1]):
            feat = X[:,idx]
            try:
                feat_binned = np.digitize(feat, bins=np.histogram_bin_edges(feat, bins=n_bins)) - 1
                joint_counts = np.histogram2d(
                    feat_binned, y, bins=[np.arange(n_bins + 1), 
                    np.arange(np.max(y) + 2)])[0]
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
        similarities = np.zeros((X.shape[1], X.shape[1]))
        for feat1 in range(X.shape[1]):
            similarities[feat1, feat1] = 1.0
            for feat2 in range(feat1 + 1, X.shape[1]):
                corr = np.abs(pearsonr(X[:,feat1], X[:,feat2]).statistic)
                similarities[feat1, feat2] = corr
                similarities[feat2, feat1] = corr

        # Sort independent indices by mutual information.
        feat_idx = np.array([i for i in range(X.shape[1])], dtype=np.int32)
        sorted_ids = np.argsort(-mut_inf).reshape(-1)
        feat_idx = feat_idx[sorted_ids]
        mut_inf_sorted = mut_inf[sorted_ids]
        keep_idx = []
        for i in feat_idx:
            if len(keep_idx) > 0:
                max_corr = np.max(similarities[i, keep_idx])
                if max_corr < self.corr_cut:
                    keep_idx.append(i)
            else:
                keep_idx.append(i)

        # Choose features for outlier detection and classification.
        self.ood_feat = keep_idx[0:self.n_outlier] if self.outlier_feat is None else self.outlier_feat
        self.keep_idx = keep_idx[0:self.n_feat] if self.class_feat is None else self.class_feat
        if self.ignore_feat > 0:
            self.keep_idx = self.keep_idx[self.ignore_feat:]

        # Compute probability distributions for each chosen feature for each class.
        self.all_feat = list(set(self.ood_feat) | set(self.keep_idx))
        self.classes = np.unique(y).astype(np.int8)
        eval_funcs = {c: {} for c in self.classes}
        for idx in self.all_feat:
            for c in self.classes:
                class_idx = np.argwhere(y == c).reshape(-1)
                v = torch.tensor(X[class_idx,idx], dtype=torch.float)
                kde = KernelDensityEstimate(v=v)
                eval_funcs[c][idx] = kde
       
        # Save values for future use and analysis.
        self.chosen_features = self.keep_idx
        self.mutual_information = mut_inf[keep_idx]
        self.mutual_information_all = mut_inf
        self.eval_funcs = eval_funcs

    def estimate(self, X, ood=True):
        '''Compute class probabilities for each entry in X.'''
        X = torch.tensor(X, dtype=torch.float)
        if ood:
            prob = torch.ones(size=(X.shape[0], len(self.all_feat), len(self.classes)))
            for i, idx in enumerate(self.all_feat):
                for c in self.classes:
                    prob[:,i,c] = self.eval_funcs[c][idx].evaluate(X[:,idx])
            return prob
        else:
            prob = torch.ones(size=(X.shape[0], len(self.keep_idx), len(self.classes)))
            for i, idx in enumerate(self.keep_idx):
                for c in self.classes:
                    prob[:,i,c] = self.eval_funcs[c][idx].evaluate(X[:,idx])
            return prob

    def get_log_prob(self, X, ood=True):
        '''Get labels for each entry in X.'''
        prob = self.estimate(X, ood=ood)
        class_log_prob = torch.sum(torch.log(prob + 1e-30), dim=1).numpy()
        return class_log_prob
        
    def predict(self, X, ood=True):
        '''Get labels for each entry in X.'''

        # Get probabilities of belonging to each class.
        prob = self.estimate(X, ood=ood)
        if ood:
            feature_mask = torch.all(prob < 5e-4, axis=2)
            instance_mask = torch.any(feature_mask, axis=1)
            ood_idx = torch.where(instance_mask)[0]

        # Get labels.
        keep_features_idx = [self.all_feat.index(i) for i in self.keep_idx]
        class_log_prob = torch.sum(torch.log(prob[:,keep_features_idx,:] + 1e-30), dim=1).numpy()
        labels = np.argmax(class_log_prob, axis=1)
        if ood:
            labels[ood_idx] = -1

        return labels
    
    def confidence(self, X, ood=True):
        '''Get log(scores) for each entry in X.'''
        prob = self.estimate(X, ood=ood)
        class_log_prob = torch.sum(torch.log(prob + 1e-30), dim=1).numpy()
        if ood:
            feature_mask = torch.all(prob < 5e-4, axis=2)
            instance_mask = torch.any(feature_mask, axis=1)
            ood_idx = torch.where(instance_mask)[0]
            class_log_prob[outlier_idx,c] = -np.inf
        return np.exp(class_log_prob)
    
if __name__ == '__main__':

    from sklearn.model_selection import KFold

    state_labels = {
        'hda': 0,
        'lda': 1,
        'ice': 2
    }

    def load_data(model, feat='all', size=16, states=['hda', 'lda']):
        ''' Method for efficiently loading environments for a given number of neighbors. '''
        desc_dir = '.'
        descs = []
        labels = []
        for state in states:
            if feat == 'stein':
                stein = np.load(f'{desc_dir}/descriptors/neigh_{size}/{model}_{state}_stein.npy')
                descs.append(stein)
                for _ in range(stein.shape[0]):
                    labels.append(state_labels[state])
            elif feat == 'acsf':
                acsf = np.load(f'{desc_dir}/descriptors/neigh_{size}/{model}_{state}_acsf.npy')
                descs.append(acsf)
                for _ in range(acsf.shape[0]):
                    labels.append(state_labels[state])
            elif feat == 'all':
                acsf = np.load(f'{desc_dir}/descriptors/neigh_{size}/{model}_{state}_acsf.npy')
                stein = np.load(f'{desc_dir}/descriptors/neigh_{size}/{model}_{state}_stein.npy')
                desc = np.hstack((acsf, stein))
                descs.append(desc)
                for _ in range(desc.shape[0]):
                    labels.append(state_labels[state])
        desc = np.vstack(descs)
        labels = np.array(labels)
        return desc, labels

    X, y = load_data(model='mbpol', feat='all')
    print(f'Number of features: {X.shape[1]}')
    hda_idx = np.argwhere(y == 0).reshape(-1)
    lda_idx = np.argwhere(y == 1).reshape(-1)

    log_prob_lda = []
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    for idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f'Evaluating fold {idx + 1} / 5.')
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]
        model = ProbabilisticModel(
            max_features=5, 
            include=0.98,
            detect_outliers=False,
            corr_cut=0.8,
            use_features=[44, 104, 105, 108, 109]
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test, binary=True)   # Only consider HDA/LDA predictions.
        log_probs = model.get_log_prob(X_test, binary=True) # Only consider HDA/LDA predictions.
        lda_idx = np.argwhere(y_test == 1).reshape(-1) 
        log_prob_lda.append(log_probs[lda_idx,1])
    log_prob_lda = np.hstack(log_prob_lda).reshape(-1)

    # Retrain our model on all HDA and LDA configurations.
    model = ProbabilisticModel(
        max_features=5, 
        include=0.98,
        detect_outliers=False,
        corr_cut=0.8,
        use_features=[44, 104, 105, 108, 109]
    )
    model.fit(X, y)

    # Evaluate trained model when extrapolating to hexagonal ice structures.
    X_ice_acsf = np.load(f'./descriptors/neigh_16/mbpol_ice_acsf.npy')
    X_ice_stein = np.load(f'./descriptors/neigh_16/mbpol_ice_stein.npy')
    X_ice = np.hstack((X_ice_acsf, X_ice_stein))

    feat = 117
    plt.hist(X_ice[:,feat], bins=30, label='Ice', alpha=0.5, density=True)
    lda_idx = np.argwhere(y == 1).reshape(-1)
    plt.hist(X[lda_idx,feat], bins=30, label='LDA', alpha=0.5, density=True)
    plt.legend()
    plt.show()

    log_prob_ice = model.get_log_prob(X_ice, binary=True)[:,1]
    plt.hist(log_prob_ice, bins=30, label='Ice', alpha=0.5, density=True)
    plt.hist(log_prob_lda, bins=30, label='LDA', alpha=0.5, density=True)
    plt.legend()
    plt.show()