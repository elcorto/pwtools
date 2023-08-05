import numpy as np
from pwtools.crys import nearest_neighbors, Structure

def aequal(a,b):
    ret = (a == b)
    if type(ret) == type(True):
        return ret
    else:
        return ret.all()

def test_nn():
    # make cell big enough to avoid pbc wrap, we only test cases where pbc=True
    # == pbc=False, we trust that pbc distances work, see crys.rpdf() etc
    cell = np.identity(3) * 10
    xcoords = np.array(\
        [ 1,   2.02, 3.1,  4,    4.9, 6.01, 7.03])
    symbols = \
        ['H', 'H',   'O', 'Ca', 'O', 'Cl', 'Cl']
    coords = np.zeros((len(xcoords),3), dtype=float)
    coords[:,0] = xcoords
    struct = Structure(coords=coords, cell=cell, symbols=symbols)
    asym = np.array(struct.symbols)

    # [2, 4, 1, 5, 0, 6]
    assert aequal(nearest_neighbors(struct, idx=3, num=2),
                  np.array([2,4]))

    # [1, 5, 0, 6]
    assert aequal(nearest_neighbors(struct, idx=3, num=2, skip='O'),
                  np.array([1,5]))

    # [1, 0]
    assert aequal(nearest_neighbors(struct, idx=3, num=2, skip=['O','Cl']),
                  np.array([1,0]))

    # [2, 4]
    assert aequal(nearest_neighbors(struct, idx=3, cutoff=1.2),
                  np.array([2,4]))

    # []
    assert aequal(nearest_neighbors(struct, idx=3, cutoff=1.2, skip='O'),
                  np.array([]))

    # [2,4,1,5]
    assert aequal(nearest_neighbors(struct, idx=3, cutoff=2.1, skip=None),
                  np.array([2,4,1,5]))

    # [1]
    assert aequal(nearest_neighbors(struct, idx=3, cutoff=2.1, skip=['O','Cl']),
                  np.array([1]))

    # [1,0], with dist
    d=nearest_neighbors(struct, idx=3, num=2, skip=['O','Cl'], fullout=True)[1]
    np.allclose(d, np.array([1.98,3.0]))

