import argparse
from ase import Atoms
from dscribe.descriptors import ACSF
from joblib import Parallel, delayed
import numpy as np
import os
import pickle

from probabilistic_model import ProbabilisticModel
from steinhardt import get_steinhardt_params

import functools
print = functools.partial(print, flush=True)

def compute_frame_length(traj):
    line_counter = 0
    step_counter = 0
    with open(traj, 'r') as f:
        for line in f:
            if 'ITEM: TIMESTEP' in line:
                step_counter += 1
            if step_counter == 2:
                break
            line_counter += 1
    return line_counter

def label_traj(frame, model, n_neigh, idx, total_frames):
    print(f'Computing for frame {idx + 1} / {total_frames}.')

    # Compute number of atoms to keep in environment.
    n_atoms = (n_neigh + 1) * 3

    # Convert frame to numpy array.
    x_lims = frame[5].strip().split()
    y_lims = frame[6].strip().split()
    z_lims = frame[7].strip().split()
    atom_ids = []
    coords = []
    for line in frame[9:]:
        vals = line.strip().split()
        atom_ids.append(int(vals[0]))
        coords.append([int(vals[1]), float(vals[2]), float(vals[3]), float(vals[4])])
    atom_ids = np.array(atom_ids)
    sorted_idx = np.argsort(atom_ids).reshape(-1)
    coords = np.array(coords)
    coords = coords[sorted_idx]

    # Compute descriptors for each oxygen.
    box_dim = np.array([
        float(x_lims[0]), float(x_lims[1]),
        float(y_lims[0]), float(y_lims[1]),
        float(z_lims[0]), float(z_lims[1])
    ])
    box_length = np.array([
        box_dim[1] - box_dim[0],
        box_dim[3] - box_dim[2],
        box_dim[5] - box_dim[4]
    ])
    types = coords[3:,0]
    coords = coords[3:,1:]
    atom_types = ['O' if i == 1 else 'H' for i in types]

    # Get distance vectors and neighbors for each oxygen.
    oxygen_ids = [id for id in range(len(types)) if types[id] == 1]
    neighbor_vectors = []
    neighbor_idx = []
    for starting_id in oxygen_ids:
        oxy_coord = coords[starting_id]
        dist_vec = coords - oxy_coord
        for dim_idx in range(3):
            dist_vec[:,dim_idx] = np.where(
                dist_vec[:,dim_idx] > 0.5 * box_length[dim_idx], 
                dist_vec[:,dim_idx] - box_length[dim_idx],
                dist_vec[:,dim_idx]
            )
            dist_vec[:,dim_idx] = np.where(
                dist_vec[:,dim_idx] < -0.5 * box_length[dim_idx], 
                dist_vec[:,dim_idx] + box_length[dim_idx],
                dist_vec[:,dim_idx]
            )
        dist = np.linalg.norm(dist_vec, axis=-1)
        chosen_molecule_idx = np.argsort(dist)[0:n_atoms]
        save_coords = dist_vec[chosen_molecule_idx].reshape(-1,3)
        neighbor_vectors.append(save_coords)
        neighbor_idx.append(chosen_molecule_idx)
    neighbor_vectors = np.array(neighbor_vectors)
    neighbor_idx = np.array(neighbor_idx, dtype=int)

    # Generate atomic environment feature vectors.
    all_feats = []
    for row in range(neighbor_vectors.shape[0]):
        types_ = types[neighbor_idx[row]]
        atom_types = ['O' if i == 1 else 'H' for i in types_]
        positions = neighbor_vectors[row]
        frame_struct = Atoms(atom_types, positions=positions, pbc=False)
        acsf = ACSF(
            species=types_,
            periodic=False,
            r_cut=5.0,
            g2_params=[[0.5, 1.0],[1.0, 1.0],[1.5, 1.0],[2.0, 1.0],[2.5, 1.0],[3.0, 1.0],[3.5, 1.0],[4.0, 1.0],[4.5, 1.0],[5.0, 1.0]],
            g3_params=[0.5, 1.0, 1.5, 2.0],
            g4_params=[[0.0, 1.0, 1.0],[0.0, 0.5, 1.0],[2.0, 1.0, 1.0],[2.0, 0.5, 1.0],[4.5, 1.0, 1.0],[4.5, 0.5, 1.0],[0.0, 1.0, -1.0],[0.0, 0.5, -1.0],[2.0, 1.0, -1.0],[2.0, 0.5, -1.0],[4.5, 1.0, -1.0],[4.5, 0.5, -1.0]],
            g5_params=[[2.0, 1.0, 1.0],[2.0, 0.5, 1.0],[4.5, 1.0, 1.0],[4.5, 0.5, 1.0],[2.0, 1.0, -1.0],[2.0, 0.5, -1.0],[4.5, 1.0, -1.0],[4.5, 0.5, -1.0]]
        )
        feat = acsf.create(frame_struct, centers=[0]).reshape(-1)
        all_feats.append(feat)
    all_feats = np.array(all_feats)

    # Isolate oxygen molecules.
    oxygen_idx = np.argwhere(types == 1).reshape(-1)
    types = types[oxygen_idx]
    atom_types = ['O' if i == 1 else 'H' for i in types]
    coords = coords[oxygen_idx]

    # Generate Steinhardt descriptors for each oxygen atom.
    shift_vector = box_dim[[0,2,4]]
    centered_frame = coords - shift_vector
    frame_struct = Atoms(atom_types, positions=centered_frame, cell=box_length, pbc=True)

    # Compute Steinhardt descriptors.
    stein = get_steinhardt_params(
        struct=frame_struct,
        cutoff_radius=5.0,
        numb_neighbours=n_neigh,
        q_l_values=[3,4,5,6,7,8,9,10,11,12],
        w_l_values=[4,6,8,10,12],
    )
    stein = stein.reshape(-1,30)
    all_feats = np.hstack((all_feats, stein))
    all_feats = all_feats.astype(np.float16)
    
    # Get labels and confidences.
    class_log_prob = model.confidence(X=all_feats, binary=True)
    class_prob = np.exp(class_log_prob)

    # Get labels and HDA/LDA occurrences.
    y_pred = model.predict(X=all_feats, binary=True)
    hda_pred = np.argwhere(y_pred == 0).reshape(-1)
    lda_pred = np.argwhere(y_pred == 1).reshape(-1)

    # Accumulate prediction information.
    pred_info = np.hstack((
        y_pred.reshape(-1,1),
        class_prob
    ))

    # Compute fractions.
    hda_frac = len(hda_pred) / y_pred.shape[0]
    lda_frac = len(lda_pred) / y_pred.shape[0]

    return (idx, hda_frac, lda_frac, pred_info, all_feats)

if __name__ == '__main__':

    # Get user input.
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename')
    parser.add_argument('--model')
    parser.add_argument('--out')
    args = parser.parse_args()

    # Parse size of atomic environment from model name.
    model_name = args.model
    model_params = model_name.split('_')
    environment_size = int(model_params[3])

    # Determine save file name.
    traj_dir = os.path.dirname(args.filename)
    cutoff = 5 if '.dump' in args.filename else 10
    traj_name = os.path.basename(args.filename)[:-cutoff]
    save_name = args.out if args.out is not None else traj_name

    # Get frame length for parallelization of analysis.
    frame_length = compute_frame_length(traj=args.filename)
    lines = open(args.filename, 'r').readlines()
    n_frames = int(len(lines) / frame_length)
    if 'decomp' in args.filename:
        n_frames -= 5 # Remove last few frames of decompressions because system explodes.

    # Load the provided model.
    model = pickle.load(open(f'../data/models/{args.model}.pkl', 'rb'))

    # Compute results for each atomic environment in parallel.
    results = Parallel(n_jobs=-1)(delayed(label_traj)(
        frame=lines[idx * frame_length : (idx + 1) * frame_length],
        model=model,
        n_neigh=environment_size,
        idx=idx,
        total_frames=n_frames) for idx in range(n_frames))
    results.sort(key=lambda x: x[0])

    # Check to make sure there is a directory for saving labels.
    if not os.path.exists(f'../data/labels/{args.model}'):
        os.makedirs(f'../data/labels/{args.model}')

    # Extract HDA and LDA fractions and save to file.
    fracs = [[result[1], result[2]] for result in results]
    fracs = np.array(fracs)
    np.save(f'../data/labels/{args.model}/{save_name}.frac.npy', fracs)

    # Save prediction information.
    pred_info = np.stack([result[3] for result in results], axis=0)
    np.save(f'../data/labels/{args.model}/{save_name}.pred.npy', pred_info)

    # Save features for entire trajectory (only for particular models...).
    if 'size_16' in model_name and '0.999' in model_name:
        features = np.stack([result[4] for result in results], axis=0)
        np.save(f'../data/labels/{args.model}/{save_name}.feat.npy', features)
