from math import radians, sin, cos
import numpy as np
from pwtools import crys, num
from pwtools.test import tools
rand = np.random.rand

def euler_rotation(phi,theta,psi):
    # http://mathworld.wolfram.com/RotationMatrix.html
    # http://mathworld.wolfram.com/EulerAngles.html
    # A = BCD
    # 1. the first rotation is by an angle phi about the z-axis using D,
    # 2. the second rotation is by an angle theta in [0,pi] about the former
    #    x-axis (now x') using C, and
    # 3. the third rotation is by an angle psi about the former z-axis (now
    #    z') using B.
    phi = radians(phi)
    theta = radians(theta)
    psi = radians(psi)
    sin_a = sin(phi)
    sin_b = sin(theta)
    sin_c = sin(psi)
    cos_a = cos(phi)
    cos_b = cos(theta)
    cos_c = cos(psi)
    D = np.array([[ cos_a,  sin_a,      0],
                  [-sin_a,  cos_a,      0],
                  [     0,      0,      1]])*1.0
    C = np.array([[     1,      0,      0],
                  [     0,  cos_b,  sin_b],
                  [     0, -sin_b,  cos_b]])*1.0
    B = np.array([[ cos_c,  sin_c,      0],
                  [-sin_c,  cos_c,      0],
                  [     0,      0,      1]])*1.0
    return np.dot(B, np.dot(C, D))


def test_methods():
    tr = crys.Trajectory(coords=rand(100,10,3),
                         cell=rand(100,3,3),
                         symbols=['H']*10)
    st = crys.Structure(coords=rand(10,3),
                        cell=rand(3,3),
                        symbols=['H']*10)

    for obj,indices in [(st, [0,1,2]), (tr, [0,0,1,2])]:
        if obj.is_traj:
            v0 = obj.coords[indices[0],indices[1],...]
            v1 = obj.coords[indices[0],indices[2],...]
            v2 = obj.coords[indices[0],indices[3],...]
        else:
            v0 = obj.coords[indices[0],...]
            v1 = obj.coords[indices[1],...]
            v2 = obj.coords[indices[2],...]
        # use eps=0 since the new system is not orthogonal, only test API
        o1 = crys.align_cart(obj, x=v1-v0, y=v2-v0, eps=0)
        o2 = crys.align_cart(obj, vecs=np.array([v0,v1,v2]), eps=0)
        o3 = crys.align_cart(obj, indices=indices, eps=0)
        tools.assert_dict_with_all_types_almost_equal(o1.__dict__,
                                                      o2.__dict__,
                                                      keys=o1.attr_lst)
        tools.assert_dict_with_all_types_almost_equal(o1.__dict__,
                                                      o3.__dict__,
                                                      keys=o1.attr_lst)


def test_correct():
    coords = rand(20,3)
    cell = rand(3,3)
    st = crys.Structure(coords=coords,
                        cell=cell,
                        symbols=['H'])
    # new cartesian coord sys, created by rotating the old E=identity(3) by 3
    # random angles
    angles = [rand()*360, rand()*180, rand()*360]
    rmat = euler_rotation(*tuple(angles))
    newcoords = np.array([np.dot(rmat, x) for x in st.coords])
    newcell = np.array([np.dot(rmat, x) for x in st.cell])
    st2 = crys.align_cart(st, cart=rmat, eps=1e-3)
    assert np.allclose(newcoords, np.dot(st.coords, rmat.T))
    assert np.allclose(newcoords, crys.coord_trans(st.coords,
                                                   old=np.identity(3),
                                                   new=rmat))
    assert np.allclose(newcell, np.dot(st.cell, rmat.T))
    assert np.allclose(newcell, crys.coord_trans(st.cell,
                                                 old=np.identity(3),
                                                 new=rmat))
    assert np.allclose(st2.coords, newcoords)
