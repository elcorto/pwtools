# mttk.py
#
# Fictitious masses for Nose-Hoover thermostat chains (NHCs) and barostat from
# the MTTK paper [1].
#
# Atomic units: All fictitious masses have dimension [mass * length**2] = [kg *
# m**2] and are converted to Hartree atomic units: mass[kg] / m0 (electron
# mass), length[m] / a0 (Bohr). We are not totally sure, but we belief that
# these are the units for qmass an bmass used in Abinit. As usual, this is not
# really documented anywhere, but we rely of the documentation for "noseinert"
# (ionmov 8), which states that "... the inertia factor WT of the Nose-Hoover
# thermostat (when ionmov=8), in atomic units of weight*length^2, that is
# (electron mass)*(Bohr)^2".
#
# [1] Glenn J. Martyna and Mark E. Tuckerman and Douglas J. Tobias and Michael
#     L. Klein, "Explicit reversible integrators for extended systems
#     dynamics", Mol. Phys. 87(5), pp. 1117, 1996

from pwtools.constants import kb, m0, a0

def add_doc(func):
    dct = {}
    dct['freq'] = \
    """freq : scalar
        Typical phonon frequency [Hz]."""
    dct['temp'] = \
    """temp : scalar
        Temperature [K]."""
    dct['nf'] = \
    """nf : scalar
        Number of degrees of freedom. Usually ndim*natoms for no constranis
        with ndim==3 (x,y,z)."""
    dct['ndim'] = \
    """ndim : scalar
        Number of dimensions. Usually ndim==3 (x,y,z)."""
    dct['nnos'] = \
    """nnos : scalar
        Length of the NH chain."""
    # Use dictionary string replacement:
    # >>> '%(lala)i %(xxx)s' %{'lala': 3, 'xxx': 'grrr'}
    # '3 grrr'
    func.__doc__ = func.__doc__ % dct 
    return func


@add_doc
def particle_nhc_masses(freq, temp, nf=None, nnos=4):
    """Fictitious masses Q_p for the particle NHC to thermostat the atoms.
    
    Abinit: qmass for ionmov 13.

    Parameters
    ----------
    %(freq)s
    %(temp)s
    %(nf)s
    %(nnos)s

    Returns
    -------
    [q1, qi ...] : list of length `nnos`
    q1 : mass for the 1st thermostat in the chain
    qi : masses for the 2nd, 3rd, ... thermostat
    """
    assert nnos >= 1, "nnos must be >= 1"
    qi = kb * temp / freq**2.0 / m0 / a0**2.0
    q1 = nf * qi
    return [q1] + [qi]*(nnos-1)


@add_doc
def barostat_nhc_masses(freq, temp, ndim=3, nnos=4):
    """Fictitious masses Q_b for the barostat NHC to thermostat the barostat.
    
    There is NO equivalent in Abinit. I think they use the particle NHC also
    for the barostat.

    Parameters
    ----------
    %(freq)s
    %(temp)s
    %(ndim)s
    %(nnos)s

    Returns
    -------
    [qb1, qbi ...] : list of length `nnos`
    qb1 : mass for the 1st thermostat in the chain
    qbi : masses for the 2nd, 3rd, ... thermostat
    """
    assert nnos >= 1, "nnos must be >= 1"
    qi = kb * temp / freq**2.0  / m0 / a0**2.0
    q1 = qi * ndim**2.0
    return [q1] + [qi]*(nnos-1)


@add_doc
def barostat_mass_w(freq, temp, nf=None, ndim=3):
    """Fictitious mass W for the barostat itself for isotropic cell
    fluctuations.
    
    Abinit: bmass for ionmov 13 + optcell 1.

    Parameters
    ----------
    %(freq)s
    %(temp)s
    %(nf)s
    %(ndim)s

    Returns
    -------
    W : barostat mass
    """
    return float(nf+ndim)*kb*temp / freq**2.0  / m0 / a0**2.0


@add_doc
def barostat_mass_wg(freq, temp, nf=None, ndim=3):
    """Fictitious mass W_g for the barostat itself for full cell fluctuations.
    
    Abinit: bmass for ionmov 13 + optcell 2.

    Parameters
    ----------
    %(freq)s
    %(temp)s
    %(nf)s
    %(ndim)s

    Returns
    -------
    W_g : barostat mass
    """
    return float(nf+ndim)*kb*temp / float(ndim) / freq**2.0  / m0 / a0**2.0

