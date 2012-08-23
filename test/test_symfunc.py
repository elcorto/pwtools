import numpy as np
from pwtools import crys
from pwtools import _flib, _fsymfunc, symfunc
rand = np.random.rand

def test_symfunc():
    #g4, g5
    coords_frac = np.random.rand(10,3)
    cell = np.identity(3)*3
    natoms = coords_frac.shape[0]
    struct = crys.Structure(coords_frac=coords_frac,
                            cell=cell)
    sf = symfunc.SymFunc(struct, precond=True)
    params_all = sf.get_default_params_all()
    sf.set_params(params_all[4], 4)
    sf.set_params(params_all[5], 5)
    
    for what in [4,5]:
        print "g%i ..." %what
        py_func = getattr(sf, 'g%i_py' %what)
        gpy = py_func()
        gf = sf.g45_f(what=what)
        assert (gpy > 0.0).any(), "reference is all zero"
        assert np.allclose(gpy, gf)
    
    # cutfunc
    dists = rand(10,10)*5
    rcut = 2.5
    assert np.allclose(_fsymfunc.cutfunc(dists, rcut),
                       symfunc.cutfunc(dists,rcut))
