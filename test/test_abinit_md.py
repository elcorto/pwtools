import numpy as np
from pwtools import parse, common, crys, num
from pwtools.crys import Trajectory

timeaxis = 0

def assrt_aae(*args, **kwargs): 
    np.testing.assert_array_almost_equal(*args, **kwargs)

def check(pp, none_attrs=[], extra_attrs=[]):
    """
    Parameters
    ----------
    pp : instance of parsing class
    none_attrs : attrs which are in self.attr_lst but are None after calling
        self.parse().
    extra_attrs : attrs which are not in self.attr_lst but for which a getter
        exists
    """
    print ">>> attr_lst" 
    for attr_name in sorted(pp.attr_lst):
        if not attr_name in none_attrs:
            print("    attr not None %s" %attr_name)
            assert getattr(pp, attr_name) is not None, "FAIL: %s None" %attr_name
        else:
            print("    attr None:    %s" %attr_name)
    print "<<< attr_lst" 
    print ">>> extra_attrs" 
    for attr_name in sorted(extra_attrs):
        if not attr_name in none_attrs:
            print("    attr not None %s" %attr_name)
            assert eval("pp.get_%s()" %attr_name) is not None, "FAIL: %s None" %attr_name
        else:
            print("    attr None:    %s" %attr_name)
    print "<<< extra_attrs" 


def check_cons(pp):
    """Check consistency of redundant attrs. This is mainly for debugging.
    
    Parameters
    ----------
    pp : parser class instance, parse() called

    Check consistency between cell and cryst_const. Assume:
    pp.timeaxis = 0
    cell.shape = (nstep,3,3)
    c1.shape = (nstep, 6)
    """
    print ">>> cryst_const cell" 
    cell = pp.cell
    cc = pp.cryst_const
    if cell is not None:
        nstep_cell = cell.shape[timeaxis]
        if cc is not None:
            n1 = nstep_cell
            n2 = cc.shape[timeaxis]
            assert n1 == n2, ("cell and cryst_const have "
                "different nstep: cell %i, cryst_const: %i" %(n1, n2))
            for ii in range(nstep_cell):
                assrt_aae(crys.cell2cc(cell[ii,...]), cc[ii,:])
    print "<<< cryst_const cell" 
    print ">>> volume"
    arr1 = pp.volume
    if arr1 is not None:
        cell = pp.cell
        if cell is not None:
            arr2 = np.array([crys.volume_cell(pp.cell[ii,...]) for ii in \
                             range(pp.cell.shape[timeaxis])])
            n1, n2 = arr1.shape[timeaxis], arr2.shape[timeaxis]
            nn = min(n1, n2)
            assrt_aae(arr1[-nn:], arr2[-nn:], decimal=4)
    print "<<< volume"


def run(cls, filename_base, none_attrs=[], extra_attrs=[]):
    """Parse file and check that all relevant attrs are not None, i.e. that the
    getter found something. 
    
    Note that we do not do verfication with reference data, that would be
    infinitely much work. Parsing all this weird files is complicated enough as
    it is. If you use this parsing machinery, verify results by yourself before
    using it for serious analyis.
    
    Only as a proof of concept, run the SCF parser on the MD output. No attr
    should be None. Then, the MD parser.
    """
    filename = 'files/abinit_md/%s' %filename_base
    print("#"*77)
    print("testing: %s" %filename)
    print("#"*77)
    common.system('gunzip %s.gz' %filename)
    print "--- scf ------------------------"
    scf = parse.AbinitSCFOutputFile(filename=filename)
    scf.parse()
    check(scf, none_attrs=[], extra_attrs=[])
    print "--- md -------------------------"
    md = cls(filename=filename)
    md.parse()
    check(md, none_attrs=none_attrs, extra_attrs=extra_attrs)
    check_cons(md)
    common.system('gzip %s' %filename)


def test_abinit_md():
    extra_attrs = [\
        ]
    
    filename = 'abi_ionmov13_optcell2.out'
    none_attrs = [\
##        'angles',
##        'cell',
##        'cryst_const',
##        'cryst_const_angles_lengths',
##        'ekin_vel',
##        'lengths',
##        'temperature',
##        'velocity',
        ]
    run(parse.AbinitVCMDOutputFile, filename, none_attrs=none_attrs,
        extra_attrs=extra_attrs)
    
    filename = 'abi_ionmov13_optcell1.out'
    none_attrs = [\
##        'angles',
##        'cell',
##        'cryst_const',
##        'cryst_const_angles_lengths',
##        'ekin_vel',
##        'lengths',
##        'temperature',
##        'velocity',
        ]
    run(parse.AbinitVCMDOutputFile, filename, none_attrs=none_attrs,
        extra_attrs=extra_attrs)

    filename = 'abi_ionmov13_optcell0.out'
    none_attrs = [\
##        'angles',
##        'cell',
##        'cryst_const',
##        'cryst_const_angles_lengths',
##        'ekin_vel',
##        'lengths',
##        'temperature',
##        'velocity',
        ]
    run(parse.AbinitVCMDOutputFile, filename, none_attrs=none_attrs,
        extra_attrs=extra_attrs)

    filename = 'abi_ionmov2_optcell2.out'
    none_attrs = [\
##        'angles',
##        'cell',
##        'cryst_const',
##        'cryst_const_angles_lengths',
        'ekin_vel',
##        'lengths',
        'temperature',
        'velocity',
        ]
    run(parse.AbinitMDOutputFile, filename, none_attrs=none_attrs,
        extra_attrs=extra_attrs)
    
    filename = 'abi_ionmov2_optcell1.out'
    none_attrs = [\
##        'angles',
##        'cell',
##        'cryst_const',
##        'cryst_const_angles_lengths',
        'ekin_vel',
##        'lengths',
        'temperature',
        'velocity',
        ]
    run(parse.AbinitMDOutputFile, filename, none_attrs=none_attrs,
        extra_attrs=extra_attrs)
    
    filename = 'abi_ionmov2_optcell0.out'
    none_attrs = [\
##        'angles',
##        'cell',
##        'cryst_const',
##        'cryst_const_angles_lengths',
        'ekin_vel',
##        'lengths',
        'temperature',
        'velocity',
        ]
    run(parse.AbinitMDOutputFile, filename, none_attrs=none_attrs,
        extra_attrs=extra_attrs)

    filename = 'abi_ionmov8_optcell0.out'
    none_attrs = [\
##        'angles',
##        'cell',
##        'cryst_const',
##        'cryst_const_angles_lengths',
##        'ekin_vel',
##        'lengths',
##        'temperature',
##        'velocity',
        ]
    run(parse.AbinitMDOutputFile, filename, none_attrs=none_attrs,
        extra_attrs=extra_attrs)

