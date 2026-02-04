# amorphous-ice

This repository contains all code and data associated with *A Local Structural Basis to Resolve Amorphous Ices* (https://www.arxiv.org/abs/2601.17488) by Gallagher, Szukalo *et al*. This includes all code associated with generating descriptors of atomic environments and training a probabilistic model, as well as the analysis conducted on systems of HDA and LDA.

## Repository Structure

```
amorphous-ice/
├── README.md
├── environment.yml           # Conda environment specification
├── requirements.txt          # Pip dependencies (alternative)
├── src/                      # Source code
│   ├── config_template.py    # Configuration template for trajectory paths
│   ├── prep_trajectory.py    # Trajectory preprocessing
│   ├── gen_descriptors.py    # Descriptor generation (ACSF, Steinhardt)
│   ├── train.py              # Model training script
│   ├── label_trajectory.py   # Trajectory labeling with trained models
│   ├── probabilistic_model.py # Probabilistic classifier implementation
│   ├── steinhardt.py         # Steinhardt parameter calculations
│   ├── boo_nn.py             # BOO neural network
│   ├── pointnet.py           # PointNet architecture
│   ├── autoencoder_gmm.py    # Autoencoder + GMM model
├── data/                     # Data files
│   ├── descriptors/          # Pre-computed descriptors (ACSF, Steinhardt)
│   │   └── neigh_{2,3,...,16}/
│   ├── frames/               # Processed trajectory frames
│   ├── models/               # Trained model files (.pkl)
│   ├── labels/               # Trajectory predictions
│   └── thermo/               # Thermodynamic data
├── notebooks/                # Analysis notebooks
│   ├── section_a.ipynb       # Model training & evaluation
│   ├── section_b.ipynb       # Phase diagram analysis
│   ├── section_c.ipynb       # Ice phase analysis
│   ├── section_d.ipynb       # Compression/decompression analysis
│   ├── section_e.ipynb       # Supplementary analysis
│   └── data/                 # Reduced data for notebooks
└── scripts/                  # Utility scripts
    └── train.sh              # Batch training script
```

## Installation

Create and activate a conda environment:

```bash
conda env create -f environment.yml
conda activate amorphous-ice
```

Alternatively, install with pip:

```bash
pip install -r requirements.txt
```

## Quick Start: Reproducing the Analysis

To reproduce the figures from the paper, the pre-computed descriptors and trained models are provided. Simply open the Jupyter notebooks in `notebooks/` and run them:

```bash
cd notebooks
jupyter notebook section_a.ipynb
```

Notebooks are organized by paper section:
- `section_a.ipynb`: Model training and cross-validation (Section III.A)
- `section_b.ipynb`: Phase diagram analysis (Section III.B)
- `section_c.ipynb`: Ice phase extrapolation (Section III.C)
- `section_d.ipynb`: Compression/decompression trajectories (Section III.D)
- `section_e.ipynb`: SCAN/MBpol analysis (Section III.E)

## Applying to Your Own Data

### 1. Processing Molecular Dynamics Trajectories

First, configure paths to your LAMMPS trajectory files:

```bash
cd src
cp config_template.py config.py
# Edit config.py to add your trajectory paths
```

Then process the trajectories:

```bash
python prep_trajectory.py
```

This reads trajectories from the paths specified in `config.py` and saves processed frames as numpy arrays in `data/frames/`.

### 2. Generating Descriptors

Generate ACSF and Steinhardt descriptors for the processed frames:

```bash
python gen_descriptors.py --model mbpol --n_neigh 16
```

Options:
- `--model`: Water model identifier (`scan` or `mbpol`) - can be updated to reflect other models
- `--n_neigh`: Number of neighboring molecules in local environment (default: 16)

Descriptors are saved in `data/descriptors/neigh_{n_neigh}/`.

### 3. Training a Probabilistic Model

Train the probabilistic classifier:

```bash
python train.py --model mbpol --size 16 --n_feat 5 --include 0.999
```

Options:
- `--model`: Water model identifier
- `--size`: Neighborhood size (must match descriptor generation)
- `--n_feat`: Number of features to select
- `--include`: Probability threshold for confident predictions

Or run the batch training script:

```bash
cd ..
bash scripts/train.sh
```

Trained models are saved in `data/models/`.

### 4. Labeling New Trajectories

Label new trajectories with a trained model:

```bash
python label_trajectory.py --filename /path/to/trajectory.dump \
                           --model model_mbpol_size_16_feat_5_include_0.999
```

Results are saved in `data/labels/{model_name}/`.

## Available Models

The repository includes implementations of several classification approaches:

| Model | File | Input | Description |
|-------|------|-------|-------------|
| Probabilistic Model | `probabilistic_model.py` | ACSF + Steinhardt | Naive Bayes with KDE (this work) |
| BOO-NN | `boo_nn.py` | Steinhardt only | https://doi.org/10.1063/5.0193340 |
| PointNet | `pointnet.py` | Raw coordinates | https://doi.org/10.1039/C9SC02097G |
| AE-GMM | `autoencoder_gmm.py` | Steinhardt only | https://doi.org/10.1063/1.5118867 |

## Citation

If you use this code, please cite:

```bibtex
@article{Gallagher:2026,
  title={A Local Structural Basis to Resolve Amorphous Ices},
  author={Gallagher, Quinn M and Szukalo, Ryan J and Giovambattista, Nicolas and Debenedetti, Pablo G and Webb, Michael A},
  journal={arXiv preprint arXiv:2601.17488},
  year={2026}
}```
