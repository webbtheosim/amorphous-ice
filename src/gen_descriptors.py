from ase import Atoms
from dscribe.descriptors import ACSF
from joblib import Parallel, delayed
import numpy as np
import os
from steinhardt import get_steinhardt_params

from config import scan_paths, mbpol_paths

import functools
print = functools.partial(print, flush=True)

def compute_acsf_and_stein(frame, idx, n_neigh):
    print(f'Preparing configuration {idx + 1}.')

    # Compute number of atoms to include in local environment.
    total_atoms = (n_neigh + 1) * 3

    # Process frame.
    box_dim = np.array([
        frame[0,2].item(),
        frame[0,3].item(),
        frame[1,2].item(),
        frame[1,3].item(),
        frame[2,2].item(),
        frame[2,3].item()
    ])
    box_length = np.array([
        box_dim[1] - box_dim[0],
        box_dim[3] - box_dim[2],
        box_dim[5] - box_dim[4]
    ])
    types = frame[3:,0]
    atom_types = ['O' if i == 1 else 'H' for i in types]
    coords = frame[3:,1:]

    # Choose an oxygen from this frame.
    oxygen_idx = [i for i in range(len(atom_types)) if atom_types[i] == 'O']
    key_idx = np.random.randint(low=0, high=len(oxygen_idx))

    # Isolate atomic environment for ACSFs.
    oxy_coord = coords[oxygen_idx[key_idx]]
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
    chosen_molecule_idx = np.argsort(dist)[0:total_atoms]
    save_coords = dist_vec[chosen_molecule_idx].reshape(-1,3)
    save_types = types[chosen_molecule_idx].reshape(-1,1)
    save_coords = np.hstack((save_types, save_coords))

    # Generate ACSF.
    idx = np.argsort(np.linalg.norm(save_coords[:,1:], axis=-1))[0:total_atoms]
    X_coords = save_coords[idx, 1:]
    X_types = save_coords[idx, 0]
    acsf_types = ['O' if i == 1 else 'H' for i in X_types]
    acsf = ACSF(
        species=acsf_types,
        periodic=False,
        r_cut=5.0,
        g2_params=[[0.5, 1.0],[1.0, 1.0],[1.5, 1.0],[2.0, 1.0],[2.5, 1.0],[3.0, 1.0],[3.5, 1.0],[4.0, 1.0],[4.5, 1.0],[5.0, 1.0]],
        g3_params=[0.5, 1.0, 1.5, 2.0],
        g4_params=[[0.0, 1.0, 1.0],[0.0, 0.5, 1.0],[2.0, 1.0, 1.0],[2.0, 0.5, 1.0],[4.5, 1.0, 1.0],[4.5, 0.5, 1.0],[0.0, 1.0, -1.0],[0.0, 0.5, -1.0],[2.0, 1.0, -1.0],[2.0, 0.5, -1.0],[4.5, 1.0, -1.0],[4.5, 0.5, -1.0]],
        g5_params=[[2.0, 1.0, 1.0],[2.0, 0.5, 1.0],[4.5, 1.0, 1.0],[4.5, 0.5, 1.0],[2.0, 1.0, -1.0],[2.0, 0.5, -1.0],[4.5, 1.0, -1.0],[4.5, 0.5, -1.0]]
    )
    frame_struct = Atoms(acsf_types, positions=X_coords, pbc=False)
    acsf_vec = acsf.create(frame_struct, centers=[0]).reshape(-1)

    # Get atomic environment for Steinhardt descriptors.
    coords = coords[oxygen_idx]
    atom_types = ['O' for _ in range(coords.shape[0])]
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
    stein_vec = stein[key_idx]

    return (save_coords, acsf_vec, stein_vec)

if __name__ == '__main__':
    np.random.seed(1)

    # Get user input for what descriptors to generate.
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['scan', 'mbpol'], default='mbpol')
    parser.add_argument('--n_neigh', type=int, default=16)
    args = parser.parse_args()

    # Specify the identifier and paths to use.
    IDENTIFIER = args.model
    N_NEIGH = args.n_neigh
    file_paths = mbpol_paths if args.model == 'mbpol' else scan_paths
    states = file_paths.keys()

    # Make sure appropriate directories are present.
    if not os.path.exists('../data/descriptors'):
        os.makedirs('../data/descriptors')
    if not os.path.exists(f'../data/descriptors/neigh_{N_NEIGH}'):
        os.mkdir(f'../data/descriptors/neigh_{N_NEIGH}')

    # Compute descriptors for each state.
    for state in states:
        print(f'Generating configurations for state: {state}')

        # Get filenames to analyze.
        filenames = [filename for filename in os.listdir('../data/frames/') if (IDENTIFIER in filename and state in filename)]

        # Choose a random set of frames from available frames; compute descriptors
        # for randomly chosen point in that frame.
        coords = []
        acsf_desc = []
        stein_desc = []
        for file_idx, filename in enumerate(filenames):
            traj = np.load(f'../data/frames/{filename}')
            frame_ids = [i for i in range(traj.shape[0])]

            # Compute descriptors for the chosen frames.
            results = Parallel(n_jobs=-1)(delayed(compute_acsf_and_stein)(
                frame=traj[frame_idx],
                idx=idx,
                n_neigh=N_NEIGH
            ) for idx, frame_idx in enumerate(frame_ids))

            # Save the Steinhardt descriptors.
            for result in results:
                coords.append(result[0])
                acsf_desc.append(result[1])
                stein_desc.append(result[2])

        # Save descriptors.
        coords = np.stack(coords, axis=0).astype(np.float32)
        np.save(f'../data/descriptors/neigh_{N_NEIGH}/{IDENTIFIER}_{state}_coords.npy', coords)

        acsf_desc = np.stack(acsf_desc, axis=0).astype(np.float32)
        np.save(f'../data/descriptors/neigh_{N_NEIGH}/{IDENTIFIER}_{state}_acsf.npy', acsf_desc)

        stein_desc = np.stack(stein_desc, axis=0).astype(np.float32)
        np.save(f'../data/descriptors/neigh_{N_NEIGH}/{IDENTIFIER}_{state}_stein.npy', stein_desc)
