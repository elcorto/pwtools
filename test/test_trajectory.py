# We assume all lengths in Angstrom. Only important for ASE comparison.
#
import types, copy
import numpy as np
from scipy.signal import hanning
from pwtools.crys import Trajectory, Structure
from pwtools import crys, constants
from pwtools.test.tools import aaae, assert_all_types_equal,\
    assert_attrs_not_none, assert_dict_with_all_types_equal
from pwtools.test.rand_container import get_rand_struct, get_rand_traj
from pwtools import num
rand = np.random.rand


def remove_from_lst(lst, items):
    _lst = copy.deepcopy(lst)
    for item in items:
        while item in _lst:
            _lst.remove(item)
        if item in _lst:
            raise StandardError("hmm, item %s still in lst" %item)
    return _lst



def test_traj():
    natoms = 10
    nstep = 100
    cell = rand(nstep,3,3)
    stress = rand(nstep,3,3)
    forces = rand(nstep,natoms,3)
    etot = rand(nstep)
    cryst_const = crys.cell2cc3d(cell, axis=0)
    coords_frac = rand(nstep,natoms,3)
    coords = crys.coord_trans3d(coords=coords_frac,
                                old=cell,
                                new=num.extend_array(np.identity(3),
                                                     nstep,axis=0),
                                axis=1,
                                timeaxis=0)                                                    
    assert cryst_const.shape == (nstep, 6)
    assert coords.shape == (nstep,natoms,3)
    symbols = ['H']*natoms
    
    # automatically calculated:
    #   coords
    #   cell
    #   pressure
    #   velocity (from coords)
    #   temperature (from ekin)
    #   ekin (from velocity)
    traj = Trajectory(coords_frac=coords_frac,
                    cell=cell,
                    symbols=symbols,
                    forces=forces,
                    stress=stress,
                    etot=etot,
                    timestep=1,
                    )
    # Test if all getters work.
    for name in traj.attr_lst:
        print "test if getters work:", name
        traj.try_set_attr(name)
        assert getattr(traj, name) is not None, "attr None: %s" %name
        assert eval('traj.get_%s()'%name) is not None, "getter returns None: %s" %name
        print "test if getters work:", name, "... ok"
    aaae(coords_frac, traj.coords_frac)
    aaae(coords, traj.coords)
    aaae(cryst_const, traj.cryst_const)
    aaae(np.trace(stress, axis1=1, axis2=2)/3.0, traj.pressure)
    assert traj.coords.shape == (nstep,natoms,3)
    assert traj.cell.shape == (nstep,3,3)
    assert traj.velocity.shape == (nstep, natoms, 3)
    assert traj.temperature.shape == (nstep,)
    assert traj.ekin.shape == (nstep,)
    assert traj.nstep == nstep
    assert traj.natoms == natoms

    traj = Trajectory(coords_frac=coords_frac,
                    symbols=symbols,
                    cell=cell)
    aaae(coords, traj.coords)
    
    # Cell calculated from cryst_const has defined orientation in space which may be
    # different from the original `cell`, but the volume and underlying cryst_const
    # must be the same.
    traj = Trajectory(coords_frac=coords_frac,
                    symbols=symbols,
                    cryst_const=cryst_const)
    np.testing.assert_almost_equal(crys.volume_cell3d(cell),
                                   crys.volume_cell3d(traj.cell))
    aaae(cryst_const, crys.cell2cc3d(traj.cell))
    
    # extend arrays
    cell2d = rand(3,3)
    cc2d = crys.cell2cc(cell2d)
    traj = Trajectory(coords_frac=coords_frac,
                      cell=cell2d,
                      symbols=symbols)
    assert traj.cell.shape == (nstep,3,3)
    assert traj.cryst_const.shape == (nstep,6)
    for ii in range(traj.nstep):
        assert (traj.cell[ii,...] == cell2d).all()
        assert (traj.cryst_const[ii,:] == cc2d).all()
    
    traj = Trajectory(coords_frac=coords_frac,
                      cryst_const=cc2d,
                      symbols=symbols)
    assert traj.cell.shape == (nstep,3,3)
    assert traj.cryst_const.shape == (nstep,6)
    for ii in range(traj.nstep):
        assert (traj.cryst_const[ii,:] == cc2d).all()

    # units
    traj = Trajectory(coords_frac=coords_frac,
                    cell=cell,
                    symbols=symbols,
                    stress=stress,
                    forces=forces,
                    units={'length': 2, 'forces': 3, 'stress': 4})
    aaae(2*coords, traj.coords)                    
    aaae(3*forces, traj.forces)                    
    aaae(4*stress, traj.stress)                    
    
    # iterate, check if Structures are complete
    traj = Trajectory(coords=coords, 
                      symbols=symbols,
                      cell=cell,
                      forces=forces,
                      stress=stress,
                      etot=etot,
                      timestep=1.0)
    for struct in traj:
        assert struct.is_struct, "st is not Structure"
        assert not struct.is_traj, "st is Trajectory"
        assert_attrs_not_none(struct)
    struct = traj[0]
    for attr_name in traj.attr_lst:
        if attr_name in struct.attrs_only_traj:
            msg = "tr[0] %s is not None" %attr_name
            assert getattr(struct,attr_name) is None, msg
        else:            
            msg = "tr[0] %s is None" %attr_name
            assert getattr(struct,attr_name) is not None, msg
    
    # slices, return traj
    keys = traj.attr_lst[:]
    tsl = traj[10:80:2]
    assert tsl.nstep == traj.nstep / 2 - 15
    assert_attrs_not_none(tsl, attr_lst=keys)
    tsl = traj[slice(10,80,2)]
    assert tsl.nstep == traj.nstep / 2 - 15
    assert_attrs_not_none(tsl, attr_lst=keys)
    tsl = traj[np.s_[10:80:2]]
    assert tsl.nstep == traj.nstep / 2 - 15
    assert_attrs_not_none(tsl, attr_lst=keys)
    assert tsl.is_traj
    
    # iteration over sliced traj
    tsl = traj[10:80:2]
    for x in tsl:
        pass
    for x in tsl.copy():
        pass

    # repeat iter
    for i in range(2):
        cnt = 0
        for st in traj:
            cnt += 1
        assert cnt == nstep, "%i, %i" %(cnt, nstep)    
    
    # copy
    traj2 = traj.copy()
    for name in traj.attr_lst:
        val = getattr(traj,name)
        if val is not None and not (isinstance(val, types.IntType) or \
            isinstance(val, types.FloatType)):
            val2 = getattr(traj2,name)
            print "test copy:", name, type(val), type(val2)
            assert id(val2) != id(val)
            assert_all_types_equal(val2, val)
    assert_dict_with_all_types_equal(traj.__dict__, traj2.__dict__,
                                     keys=traj.attr_lst)


def test_concatenate():
    st = get_rand_struct()
    nstep = 5
    
    # cat Structure
    tr_cat = crys.concatenate([st]*nstep)
    keys = tr_cat.attr_lst
    assert tr_cat.nstep == nstep
    for x in tr_cat:
        assert_dict_with_all_types_equal(x.__dict__, st.__dict__,
                                         keys=keys)
    none_attrs = ['ekin', 'timestep', 'velocity', 'temperature', 'time']
    for attr_name in tr_cat.attrs_nstep:
        if attr_name in none_attrs:
            assert getattr(tr_cat, attr_name) is None
        else:
            print "test_concatenate: shape[0] == nstep:", attr_name
            assert getattr(tr_cat, attr_name).shape[0] == nstep
            print "test_concatenate: shape[0] == nstep:", attr_name, "...ok"
    
    # cat Trajectory
    tr = get_rand_traj()
    tr_cat = crys.concatenate([tr]*3)
    assert tr_cat.nstep == 3*tr.nstep
    none_attrs = ['timestep', 'time']
    keys = remove_from_lst(tr_cat.attr_lst, none_attrs)
    for x in [tr_cat[0:tr.nstep], 
              tr_cat[tr.nstep:2*tr.nstep], 
              tr_cat[2*tr.nstep:3*tr.nstep]]:
        assert_dict_with_all_types_equal(x.__dict__, tr.__dict__,
                                         keys=keys) 
    for attr_name in tr_cat.attrs_nstep:
        if attr_name in none_attrs:
            assert getattr(tr_cat, attr_name) is None
        else:
            assert getattr(tr_cat, attr_name).shape[0] == 3*tr.nstep

    # cat mixed, Structure is minimal API
    st = get_rand_struct()
    tr = crys.concatenate([st]*5)
    tr_cat = crys.concatenate([st]*5 + [tr])
    assert tr_cat.nstep == 10
    none_attrs = ['ekin', 'timestep', 'velocity', 'temperature', 'time']
    keys = remove_from_lst(tr_cat.attr_lst, none_attrs)
    for x in tr_cat:
        assert_dict_with_all_types_equal(x.__dict__, st.__dict__,
                                         keys=keys)
    for attr_name in tr_cat.attrs_nstep:
        if attr_name in none_attrs:
            assert getattr(tr_cat, attr_name) is None
        else:
            assert getattr(tr_cat, attr_name).shape[0] == 10


def test_populated_attrs():
    class Dummy(object):
        def __init__(self, **kwds):
            self.attr_lst = ['a','b','c', 'd']
            for name in self.attr_lst:
                setattr(self, name, None)
            for k,v in kwds.iteritems():
                setattr(self, k, v)
    
    x = Dummy(a=1,b=2,c=3,d=8)
    y = Dummy(a=4)
    z = Dummy(d=7)
    assert crys.populated_attrs([x,y]) == set(['a'])
    assert crys.populated_attrs([x,y,z]) == set([])


def test_api():
    tr = get_rand_traj()
    st = get_rand_struct()
    for name in st.attr_lst:
        assert getattr(tr, name) is not None
    for name in tr.attrs_only_traj:
        assert getattr(st, name) is None
    
    aa = tr[0]      # Structure
    bb = tr[0:1]    # Trajectory
    keys = set.difference(set(aa.attr_lst), set(aa.attrs_only_traj))
    assert aa.is_struct
    assert bb.is_traj
    # remove timeaxis before comparing arrays
    for name in bb.attrs_nstep:
        attr = getattr(bb, name)
        if attr.ndim == 1:
            setattr(bb, name, attr[0])
        else:            
            setattr(bb, name, attr[0,...])
    assert_dict_with_all_types_equal(aa.__dict__, bb.__dict__, keys=keys)


def test_mean():
    tr = get_rand_traj()
    st_mean = crys.mean(tr)
    attrs_only_traj = ['time', 'timestep', 'nstep']
    for attr_name in tr.attr_lst:
        attr = getattr(st_mean, attr_name)
        if attr_name in attrs_only_traj:
            assert attr is None, "%s is not None" %attr_name
        elif attr_name in tr.attrs_nstep:
            assert np.allclose(attr, 
                               getattr(tr, attr_name).mean(axis=tr.timeaxis))
        

def test_smooth():
    tr = get_rand_traj()
    assert len(tr.attrs_nstep) > 0
    trs = crys.smooth(tr, hanning(11))
    assert len(trs.attrs_nstep) > 0
    assert_attrs_not_none(trs, attr_lst=tr.attr_lst)
    for name in tr.attrs_nstep:
        a1 = getattr(tr, name)
        a2 = getattr(trs, name)
        assert a1.shape == a2.shape
        assert np.abs(a1 - a2).sum() > 0.0
    assert trs.timestep == tr.timestep
    assert trs.nstep == tr.nstep
    
    # reproduce data with kernel [0,1,0]
    trs = crys.smooth(tr, hanning(3))
    for name in tr.attrs_nstep:
        a1 = getattr(tr, name)
        a2 = getattr(trs, name)
        assert np.allclose(a1, a2)
    
    trs1 = crys.smooth(tr, hanning(3), method=1)
    trs2 = crys.smooth(tr, hanning(3), method=2)
    assert len(trs1.attrs_nstep) > 0
    assert len(trs2.attrs_nstep) > 0
    for name in tr.attrs_nstep:
        a1 = getattr(tr, name)
        a2 = getattr(trs1, name)
        a3 = getattr(trs2, name)
        assert np.allclose(a1, a2)
        assert np.allclose(a1, a3)
    
    trs1 = crys.smooth(tr, hanning(11), method=1)
    trs2 = crys.smooth(tr, hanning(11), method=2)
    assert len(trs1.attrs_nstep) > 0
    assert len(trs2.attrs_nstep) > 0
    for name in trs1.attrs_nstep:
        a1 = getattr(trs1, name)
        a2 = getattr(trs2, name)
        assert np.allclose(a1, a2)


def test_coords_trans():
    natoms = 10
    nstep = 100
    cell = rand(nstep,3,3)
    cryst_const = crys.cell2cc3d(cell, axis=0)
    coords_frac = rand(nstep,natoms,3)
    coords = crys.coord_trans3d(coords=coords_frac,
                                old=cell,
                                new=num.extend_array(np.identity(3),
                                                     nstep,axis=0),
                                axis=1,
                                timeaxis=0)                                                    
    
    traj = Trajectory(coords_frac=coords_frac,
                      cell=cell)
    assert np.allclose(cryst_const, traj.cryst_const)
    assert np.allclose(coords, traj.coords)
    
    traj = Trajectory(coords=coords,
                      cell=cell)
    assert np.allclose(coords_frac, traj.coords_frac)


def test_compress():
    old = get_rand_traj()
    # fake integer arrary which should not be casted
    assert old.forces is not None
    old.forces = np.ones_like(old.forces).astype(int)
    float_dtype_old = old.coords.dtype
    float_dtype_new = np.float32
    assert float_dtype_new != float_dtype_old
    arr_t = type(np.array([1.0]))
    forget = ['stress', 'velocity']
    new = crys.compress(old, copy=True, dtype=float_dtype_new, forget=forget)
    for name in forget:
        assert getattr(new, name) is None
        assert getattr(old, name) is not None
    for name in old.attr_lst:
        if name in forget:
            continue
        attr_old = getattr(old, name)
        if type(attr_old) == arr_t:
            attr_new = getattr(new, name)
            if (attr_old.dtype == float_dtype_old) and \
                (attr_old.dtype.kind=='f'):
                print name
                assert type(attr_new) == arr_t
                assert attr_new.dtype == float_dtype_new
                # for all non-integer attrs, there must be a small numerical
                # difference
                assert abs(attr_old - attr_new).sum() > 0.0
            else:
                assert attr_old.dtype == attr_new.dtype
    # sanity check
    assert new.forces.dtype.kind in ('u','i')
