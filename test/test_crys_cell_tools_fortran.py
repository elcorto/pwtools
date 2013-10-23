import numpy as np
from pwtools import _flib, crys, timer, num
rand = np.random.rand

def test_solve():
    n = 5
    A = rand(n,n)
    b = rand(n)
    bold = b.copy()
    assert np.allclose(np.linalg.solve(A,b), _flib.solve(A,b))
    assert (b == bold).all()


def test_frac2cart():
    coords_frac = rand(20,3)
    coords_frac_copy = coords_frac.copy()
    cell = rand(3,3)
    c1 = np.dot(coords_frac, cell)
    c2 = _flib.frac2cart(coords_frac, cell)
    c3 = crys.coord_trans(coords_frac, old=cell, new=np.identity(3))
    assert (coords_frac == coords_frac_copy).all()
    assert np.allclose(c1, c2)
    assert np.allclose(c1, c3)
    assert c2.flags.f_contiguous


def test_cart2frac():
    coords = rand(20,3)
    coords_copy = coords.copy()
    cell = rand(3,3)
    c1 = np.dot(coords, np.linalg.inv(cell)) 
    c2 = np.linalg.solve(cell.T, coords.T).T
    c3 = _flib.cart2frac(coords, cell)
    c4 = crys.coord_trans(coords, new=cell, old=np.identity(3))
    assert (coords == coords_copy).all()
    assert np.allclose(c1, c2)
    assert np.allclose(c1, c3)
    assert np.allclose(c1, c4)
    assert c3.flags.f_contiguous


def test_frac2cart_traj():
    nstep = 100
    coords_frac = rand(nstep,20,3)
    coords_frac_copy = coords_frac.copy()
    cell = rand(nstep,3,3)
    c1 = np.array([np.dot(coords_frac[ii,...], cell[ii,...]) for ii in \
                   range(nstep)])
    c2 = _flib.frac2cart_traj(coords_frac, cell)
    c3 = crys.coord_trans3d(coords_frac, old=cell, 
                            new=num.extend_array(np.identity(3), 
                                                 nstep=nstep, axis=0))
    assert (coords_frac == coords_frac_copy).all()
    assert np.allclose(c1, c2)
    assert np.allclose(c1, c3)
    assert c2.flags.f_contiguous


def test_cart2frac_traj():
    nstep = 100
    coords = rand(nstep,20,3)
    coords_copy = coords.copy()
    cell = rand(nstep,3,3)
    c1 = np.array([np.dot(coords[ii,...], np.linalg.inv(cell[ii,...])) for ii in \
                   range(nstep)])
    c2 = _flib.cart2frac_traj(coords, cell)
    c3 = crys.coord_trans3d(coords, new=cell, 
                            old=num.extend_array(np.identity(3), 
                                                 nstep=nstep, axis=0))
    assert (coords == coords_copy).all()
    assert np.allclose(c1, c2)
    assert np.allclose(c1, c3)
    assert c2.flags.f_contiguous


