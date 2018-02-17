import numpy as np
from pwtools.crys import center_on_atom
import pwtools.test.rand_container as rc

def test():
    # Explicit code duplication even though we could use [...,0,:] which should
    # work for (natoms,3) and (nstep,natoms,3) arrays [we use it in
    # center_on_atom()],  but check if it really does.
    st = rc.get_rand_struct()
    stc = center_on_atom(st, idx=0, copy=True)
    assert (st.coords_frac != stc.coords_frac).all()
    assert (st.coords != stc.coords).all()
    assert (st.coords_frac[0,:] != np.array([0.5]*3)).all()
    assert (stc.coords_frac[0,:] == np.array([0.5]*3)).all()
    assert (stc.coords_frac[1:,:] != np.array([0.5]*3)).all()

    tr = rc.get_rand_traj()
    trc = center_on_atom(tr, idx=0, copy=True)
    assert (tr.coords_frac != trc.coords_frac).all()
    assert (tr.coords != trc.coords).all()
    assert (tr.coords_frac[:,0,:] != np.array([0.5]*3)).all()
    assert (trc.coords_frac[:,0,:] == np.array([0.5]*3)).all()
    assert (trc.coords_frac[:,1:,:] != np.array([0.5]*3)).all()
