from itertools import permutations
import numpy as np
from pwtools import crys,io

def angles(struct, pbc=False):
    """Python implementation of angles."""
    nang = struct.natoms*(struct.natoms-1)*(struct.natoms-2)
    norm = np.linalg.norm
    angles = np.empty((nang,), dtype=float)
    angleidx = np.array([x for x in permutations(range(struct.natoms),3)])
    for idx,ijk in enumerate(angleidx):
        ci = struct.coords_frac[ijk[0],:]
        cj = struct.coords_frac[ijk[1],:]
        ck = struct.coords_frac[ijk[2],:]
        dij = ci - cj
        dik = ci - ck
        if pbc:
            dij = np.dot(crys.min_image_convention(dij), struct.cell)
            dik = np.dot(crys.min_image_convention(dik), struct.cell)
        else:            
            dij = np.dot(dij, struct.cell)
            dik = np.dot(dik, struct.cell)
        cang = np.dot(dij, dik) / norm(dij) / norm(dik)
        ceps = 1.0-2.2e-16
        if cang > ceps:
            cang = 1.0
        elif cang < -ceps:
            cang = -1.0
        ang = np.arccos(cang) * 180.0 / np.pi
        angles[idx] = ang
    return angles, angleidx        

def _assert(agf, agpy, aif, aipy, nang):
    eps = np.finfo(float).eps*5
    assert np.allclose(agf, agpy)
    assert (aif == aipy).all()
    assert aif.shape[0] == nang
    assert not np.isnan(agpy).any(), "python angle nan"
    assert not np.isnan(agf).any(), "fortran angle nan"
    # do we have 0 and 180 degrees?
    assert (agf < eps).any(), "no zero degree cases"
    assert (agf - 180.0 < eps).any(), "no 180 degree cases"
    assert (agf >= 0.0).all(), "negative angles"

def test_angle():
    # CaCl struct, the supercell will have 0 and 180 degrees -> check corner
    # cases
    st = io.read_cif('files/angle/rs.cif')
    st = crys.scell(st, (2,1,1))
    nang = st.natoms*(st.natoms-1)*(st.natoms-2)
    for pbc in [True,False]:
        agf, aif = crys.angles(st, pbc=pbc)
        agpy, aipy = angles(st, pbc=pbc)
        _assert(agf, agpy, aif, aipy, nang)
        print agf
