"""
Configuration template for trajectory preprocessing.

SETUP INSTRUCTIONS:
1. Copy this file to config.py:
      cp config_template.py config.py
2. Fill in paths to your LAMMPS trajectory files (.dump format)
3. Run prep_trajectory.py to process trajectories into numpy arrays

Each trajectory entry has the format:
    ['/path/to/trajectory.dump', start_frame, end_frame]

    - start_frame: First frame to include (-1 for beginning of file)
    - end_frame: Last frame to include (-1 for end of file)

CURRENT EXPECTED DIRECTORY STRUCTURE:
    TrainingData/
    ├── DP_SCAN/                    # SCAN functional trajectories
    │   ├── HDA/                    # High-density amorphous ice
    │   │   └── *.dump
    │   ├── LDA/                    # Low-density amorphous ice
    │   │   └── *.dump
    │   └── IceIh/                  # Hexagonal ice (optional)
    │       └── *.dump
    └── DP_MBpol/                   # MB-pol trajectories (same structure)

TRAJECTORY FORMAT:
    LAMMPS dump files with atom positions. The preprocessing scripts
    expect water molecules and will identify oxygen atoms to define
    molecular centers.

NOTE:
    If you are only reproducing the analysis from the paper, you do not
    need to configure this file. Pre-computed descriptors and trained
    models are provided in the repository.
"""

scan_paths = {
    'hda': [
        # ['/path/to/TrainingData/DP_SCAN/HDA/hda_trajectory_1.dump', -1, -1],
        # ['/path/to/TrainingData/DP_SCAN/HDA/hda_trajectory_2.dump', -1, -1],
    ],
    'lda': [
        # ['/path/to/TrainingData/DP_SCAN/LDA/lda_trajectory_1.dump', -1, -1],
    ],
    'ice': [
        # ['/path/to/TrainingData/DP_SCAN/IceIh/ice_ih_100K.dump', -1, -1],
    ],
    # Additional ice polymorphs (all use triclinic LAMMPS boxes):
    # 'ice01c': [['/path/to/organized/ice01c/192atoms/trajectory.dump', -1, -1]],
    # 'ice02':  [['/path/to/organized/ice02/288atoms/trajectory.dump',  -1, -1]],
    # 'ice03':  [['/path/to/organized/ice03/288atoms/trajectory.dump',  -1, -1]],
    # 'ice04':  [['/path/to/organized/ice04/384atoms/trajectory.dump',  -1, -1]],
    # 'ice05':  [['/path/to/organized/ice05/168atoms/trajectory.dump',  -1, -1]],
    # 'ice06':  [['/path/to/organized/ice06/240atoms/trajectory.dump',  -1, -1]],
    # 'ice07':  [['/path/to/organized/ice07/384atoms/trajectory.dump',  -1, -1]],
    # 'ice08':  [['/path/to/organized/ice08/192atoms/trajectory.dump',  -1, -1]],
    # 'ice09':  [['/path/to/organized/ice09/288atoms/trajectory.dump',  -1, -1]],
    # 'ice11':  [['/path/to/organized/ice11/384atoms/trajectory.dump',  -1, -1]],
    # 'ice12':  [['/path/to/organized/ice12/288atoms/trajectory.dump',  -1, -1]],
    # 'ice13':  [['/path/to/organized/ice13/168atoms/trajectory.dump',  -1, -1]],
    # 'ice14':  [['/path/to/organized/ice14/288atoms/trajectory.dump',  -1, -1]],
    # 'ice15':  [['/path/to/organized/ice15/240atoms/trajectory.dump',  -1, -1]],
}

mbpol_paths = {
    'hda': [
        # High-density amorphous ice configurations
    ],
    'lda': [
        # Low-density amorphous ice configurations
    ],
    'ice': [
        # Hexagonal ice configurations (optional)
    ],
    # Additional ice polymorphs (mix of orthogonal and triclinic boxes):
    'ice01c': [['/path/to/organized/ice01c/648atoms/trajectory.dump',  -1, -1]],
    'ice02':  [['/path/to/organized/ice02/2880atoms/trajectory.dump',  -1, -1]],
    'ice03':  [['/path/to/organized/ice03/972atoms/trajectory.dump',   -1, -1]],
    'ice04':  [['/path/to/organized/ice04/3072atoms/trajectory.dump',  -1, -1]],
    'ice05':  [['/path/to/organized/ice05/2268atoms/trajectory.dump',  -1, -1]],
    'ice06':  [['/path/to/organized/ice06/1920atoms/trajectory.dump',  -1, -1]],
    'ice07':  [['/path/to/organized/ice07/1296atoms/trajectory.dump',  -1, -1]],
    'ice08':  [['/path/to/organized/ice08/1536atoms/trajectory.dump',  -1, -1]],
    'ice09':  [['/path/to/organized/ice09/972atoms/trajectory.dump',   -1, -1]],
    'ice11':  [['/path/to/organized/ice11/1296atoms/trajectory.dump',  -1, -1]],
    'ice12':  [['/path/to/organized/ice12/1944atoms/trajectory.dump',  -1, -1]],
    'ice13':  [['/path/to/organized/ice13/2268atoms/trajectory.dump',  -1, -1]],
    'ice14':  [['/path/to/organized/ice14/1620atoms/trajectory.dump',  -1, -1]],
    'ice15':  [['/path/to/organized/ice15/1920atoms/trajectory.dump',  -1, -1]],
}
