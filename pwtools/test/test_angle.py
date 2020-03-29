from itertools import permutations
import numpy as np
from pwtools import crys,io
from pwtools.test import tools

def angles(struct, pbc=False, mask_val=999.0, deg=True):
    """Python implementation of angles."""
    nang = struct.natoms*(struct.natoms-1)*(struct.natoms-2)
    norm = np.linalg.norm
    anglesijk = np.ones((struct.natoms,)*3, dtype=float)*mask_val
    angleidx = np.array([x for x in permutations(list(range(struct.natoms)),3)])
    for ijk in angleidx:
        ii,jj,kk = ijk
        ci = struct.coords_frac[ii,:]
        cj = struct.coords_frac[jj,:]
        ck = struct.coords_frac[kk,:]
        dvij = ci - cj
        dvik = ci - ck
        if pbc:
            dvij = np.dot(crys.min_image_convention(dvij), struct.cell)
            dvik = np.dot(crys.min_image_convention(dvik), struct.cell)
        else:
            dvij = np.dot(dvij, struct.cell)
            dvik = np.dot(dvik, struct.cell)
        cang = np.dot(dvij, dvik) / norm(dvij) / norm(dvik)
        ceps = 1.0-2.2e-16
        if cang > ceps:
            cang = 1.0
        elif cang < -ceps:
            cang = -1.0
        if deg:
            anglesijk[ii,jj,kk] = np.arccos(cang) * 180.0 / np.pi
        else:
            anglesijk[ii,jj,kk] = cang
    return anglesijk, angleidx


def test_angle():
    # CaCl struct, the supercell will have 0 and 180 degrees -> check corner
    # cases
    tools.skip_if_pkg_missing('CifFile')
    st = io.read_cif('files/angle/rs.cif')
    st = crys.scell(st, (2,1,1))
    nang = st.natoms*(st.natoms-1)*(st.natoms-2)
    mask_val = 999.0
    for deg in [True,False]:
        for pbc in [True,False]:
            agf = crys.angles(st, pbc=pbc, mask_val=mask_val)
            agpy, aipy = angles(st, pbc=pbc, mask_val=mask_val)
            eps = np.finfo(float).eps*5
            assert np.allclose(agf, agpy)
            assert aipy.shape[0] == nang
            assert len((agf != mask_val).nonzero()[0]) == nang
            angleidx = np.array(list(zip(*(agf != mask_val).nonzero())))
            assert (angleidx == aipy).all()
            assert not np.isnan(agpy).any(), "python angle nan"
            assert not np.isnan(agf).any(), "fortran angle nan"
            # do we have 0 and 180 degrees?
            assert (agf < eps).any(), "no zero degree cases"
            assert (agf - 180.0 < eps).any(), "no 180 degree cases"
            assert (agf >= 0.0).all(), "negative angles"

