import kmedoids
import numpy as np
from scipy.stats import pearsonr

def load_descriptors():
    descriptors = []
    for size in [16, 12, 8, 7, 6, 5, 4, 3, 2]:
        for state in ['hda', 'lda', 'liquid', 'ice']:
            acsf = np.load(f'./neigh_{size}/scan_{state}_acsf.npy')
            stein = np.load(f'./neigh_{size}/scan_{state}_stein.npy')
            data = np.hstack((acsf, stein))
            descriptors.append(data)
        for state in ['hda', 'lda', 'liquid', 'ice']:
            acsf = np.load(f'./neigh_{size}/mbpol_{state}_acsf.npy')
            stein = np.load(f'./neigh_{size}/mbpol_{state}_stein.npy')
            data = np.hstack((acsf, stein))
            descriptors.append(data)
    descriptors = np.vstack(descriptors)
    return descriptors

if __name__ == '__main__':

    # Compute correlations between each feature.
    descriptors = load_descriptors()
    similarities = np.zeros((descriptors.shape[1], descriptors.shape[1]))
    for feat1 in range(descriptors.shape[1]):
        similarities[feat1, feat1] = 1.0
        for feat2 in range(feat1 + 1, descriptors.shape[1]):
            corr = pearsonr(descriptors[:,feat1], descriptors[:,feat2]).statistic
            similarities[feat1, feat2] = corr
            similarities[feat2, feat1] = corr

    print(np.argwhere(similarities[105] > 0.8).reshape(-1))

    # Choose a set of features which contain unique information.
    CUTOFF = 0.8
    unique_features = [0]
    for feat_idx in range(1,similarities.shape[0]):
        feat_sims = similarities[feat_idx, unique_features]
        if np.max(feat_sims) < CUTOFF:
            unique_features.append(feat_idx)

    print(unique_features)
