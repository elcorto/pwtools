import numpy as np
from math import pi, cos
from pwtools.pwscf import ibrav2cell
from pwtools import crys

def test_ibrav():
    # bogus
    aa = 3.0
    bb = 4.0
    cc = 5.0
    alpha = 66*pi/180
    beta = 77*pi/180
    gamma = 88*pi/180

    ibrav = 1
    celldm = [aa] + [None]*5
    ibrav2cell(ibrav, celldm)

    ibrav = 2
    celldm = [aa] + [None]*5
    ibrav2cell(ibrav, celldm)

    ibrav = 3
    celldm = [aa] + [None]*5
    ibrav2cell(ibrav, celldm)

    ibrav = 4
    celldm = [aa, None, cc/aa, None, None, None]
    ibrav2cell(ibrav, celldm)

    ibrav = 5
    celldm = [aa, None, None, cos(alpha), None, None]
    ibrav2cell(ibrav, celldm)

    ibrav = 6
    celldm = [aa, None, cc/aa, None, None, None]
    ibrav2cell(ibrav, celldm)

    ibrav = 7
    celldm = [aa, None, cc/aa, None, None, None]
    ibrav2cell(ibrav, celldm)

    ibrav = 8
    celldm = [aa, bb/aa, cc/aa, None, None, None]
    ibrav2cell(ibrav, celldm)

    ibrav = 9
    celldm = [aa, bb/aa, cc/aa, None, None, None]
    ibrav2cell(ibrav, celldm)

    ibrav = 10
    celldm = [aa, bb/aa, cc/aa, None, None, None]
    ibrav2cell(ibrav, celldm)

    ibrav = 11
    celldm = [aa, bb/aa, cc/aa, None, None, None]
    ibrav2cell(ibrav, celldm)

    # celldm(4)=cos(ab) in doc!?
    ibrav = 12
##    celldm = [aa, bb/aa, cc/aa, cos(alpha), None, None]
    celldm = [aa, bb/aa, cc/aa, None, None, cos(gamma)]
    ibrav2cell(ibrav, celldm)

    # celldm(4)=cos(ab) in doc!?
    ibrav = 13
##    celldm = [aa, bb/aa, cc/aa, cos(alpha), None, None]
    celldm = [aa, bb/aa, cc/aa, None, None, cos(gamma)]
    ibrav2cell(ibrav, celldm)

    # WOHOO!
    ibrav = 14
    celldm = [aa, bb/aa, cc/aa, cos(alpha), cos(beta), cos(gamma)]
    cell=ibrav2cell(ibrav, celldm)
    np.testing.assert_array_almost_equal(cell,
                                         crys.cc2cell(crys.celldm2cc(celldm)))
