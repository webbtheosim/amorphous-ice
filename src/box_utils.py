import numpy as np


def parse_lammps_box(bounds_header, x_line, y_line, z_line):
    """Parse LAMMPS dump box bounds into origin and 3x3 cell matrix.

    Parameters
    ----------
    bounds_header : str
        The 'ITEM: BOX BOUNDS ...' line
    x_line, y_line, z_line : str
        The three data lines following the bounds header

    Returns
    -------
    origin : np.ndarray, shape (3,)
        [xlo, ylo, zlo] — corner of the simulation box
    cell : np.ndarray, shape (3, 3)
        Cell matrix with lattice vectors as rows (LAMMPS convention):
        [[lx,  0,  0],
         [xy, ly,  0],
         [xz, yz, lz]]
    """
    triclinic = 'xy xz yz' in bounds_header

    x_vals = [float(v) for v in x_line.strip().split()]
    y_vals = [float(v) for v in y_line.strip().split()]
    z_vals = [float(v) for v in z_line.strip().split()]

    if triclinic:
        xlo_bound, xhi_bound, xy = x_vals
        ylo_bound, yhi_bound, xz = y_vals
        zlo_bound, zhi_bound, yz = z_vals

        # Recover true box edge lengths by un-expanding the bounding box.
        # LAMMPS enlarges the reported bounds to contain the tilted cell:
        #   xlo_bound = xlo + min(0, xy, xz, xy+xz)
        #   xhi_bound = xhi + max(0, xy, xz, xy+xz)
        #   ylo_bound = ylo + min(0, yz)
        #   yhi_bound = yhi + max(0, yz)
        xlo = xlo_bound - min(0.0, xy, xz, xy + xz)
        xhi = xhi_bound - max(0.0, xy, xz, xy + xz)
        ylo = ylo_bound - min(0.0, yz)
        yhi = yhi_bound - max(0.0, yz)
        zlo = zlo_bound
        zhi = zhi_bound
    else:
        xlo, xhi = x_vals[0], x_vals[1]
        ylo, yhi = y_vals[0], y_vals[1]
        zlo, zhi = z_vals[0], z_vals[1]
        xy = xz = yz = 0.0

    lx = xhi - xlo
    ly = yhi - ylo
    lz = zhi - zlo

    origin = np.array([xlo, ylo, zlo])
    cell = np.array([
        [lx,  0.0, 0.0],
        [xy,  ly,  0.0],
        [xz,  yz,  lz ],
    ])

    return origin, cell


def cell_from_frame(frame):
    """Reconstruct origin and cell matrix from a preprocessed frame array.

    Reads the box metadata stored by prep_trajectory.py in the first three
    rows of the frame array:
        frame[0] = [xy, xz, xlo, xhi]
        frame[1] = [yz, 0., ylo, yhi]
        frame[2] = [0., 0., zlo, zhi]

    Returns
    -------
    origin : np.ndarray, shape (3,)
    cell   : np.ndarray, shape (3, 3)
    """
    xy  = frame[0, 0].item()
    xz  = frame[0, 1].item()
    yz  = frame[1, 0].item()
    xlo = frame[0, 2].item()
    xhi = frame[0, 3].item()
    ylo = frame[1, 2].item()
    yhi = frame[1, 3].item()
    zlo = frame[2, 2].item()
    zhi = frame[2, 3].item()

    lx = xhi - xlo
    ly = yhi - ylo
    lz = zhi - zlo

    origin = np.array([xlo, ylo, zlo])
    cell = np.array([
        [lx,  0.0, 0.0],
        [xy,  ly,  0.0],
        [xz,  yz,  lz ],
    ])

    return origin, cell


def minimum_image(dist_vec, cell):
    """Apply minimum image convention for a general (possibly triclinic) cell.

    Parameters
    ----------
    dist_vec : np.ndarray, shape (N, 3)
        Displacement vectors in Cartesian coordinates
    cell : np.ndarray, shape (3, 3)
        Cell matrix with lattice vectors as rows

    Returns
    -------
    np.ndarray, shape (N, 3)
        Displacement vectors wrapped to the nearest image
    """
    inv_cell = np.linalg.inv(cell)
    frac = dist_vec @ inv_cell
    frac -= np.round(frac)
    return frac @ cell
