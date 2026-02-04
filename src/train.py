import numpy as np
import pickle

from probabilistic_model import ProbabilisticModel

state_labels = {
    'hda': 0,
    'lda': 1,
    'liquid': 2,
    'ice': 3
}

def load_data(model, size, states=['hda', 'lda']):
    ''' Method for efficiently loading environments for a given number of neighbors. '''
    desc_dir = '../data'
    descs = []
    labels = []
    for state in states:
        acsf = np.load(f'{desc_dir}/descriptors/neigh_{size}/{model}_{state}_acsf.npy')
        stein = np.load(f'{desc_dir}/descriptors/neigh_{size}/{model}_{state}_stein.npy')
        desc = np.hstack((acsf, stein))
        descs.append(desc)
        for _ in range(desc.shape[0]):
            labels.append(state_labels[state])
    desc = np.vstack(descs)
    labels = np.array(labels)
    return desc, labels

if __name__ == '__main__':

    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['scan', 'mbpol'], default='mbpol')
    parser.add_argument('--size', type=int, choices=[16, 12, 8, 7, 6, 5, 4, 3, 2], default=16)
    parser.add_argument('--n_feat', type=int, default=5)
    parser.add_argument('--include', type=float, default=0.98)
    args = parser.parse_args()

    model_name = f'model_{args.model}_size_{args.size}_feat_{args.n_feat}_include_{args.include}.pkl'
    print(f'Training model: {model_name}')

    X, y = load_data(model=args.model, size=args.size, states=['hda', 'lda'])
    model = ProbabilisticModel(
        max_features=args.n_feat,
        include=args.include,
        detect_outliers=True,
        # use_features=[0, 109, 90, 91, 108]
    )
    model.fit(X, y)

    if not os.path.exists('../data/models'):
        os.makedirs('../data/models')
    pickle.dump(model, open(f'../data/models/{model_name}', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
