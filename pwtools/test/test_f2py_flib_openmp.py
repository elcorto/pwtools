def test_f2py_flib_openmp():
    import numpy as np
    from pwtools import _flib
    from pwtools.pydos import fvacf
    from pwtools.common import backtick
    import os
    import sys

    OMP_DCT = {'num_threads': None}
    def omp_num_threads(action='check', num=1, omp_dct=OMP_DCT, err=False):
        key = 'OMP_NUM_THREADS'
        has_key = key in os.environ
        if action == 'check':
            if has_key:
                print("[omp_num_threads] os.environ['%s']: %s" %(key, os.environ[key]))
                print("[omp_num_threads] shell$ echo %s" %(key))
                print(backtick('echo $%s' %key))
                if err and os.environ[key] != '3':
                    return 'err'
            else:
                print("[omp_num_threads] no os.environ['%s']" %key)
                if err:
                    return 'err'
        elif action == 'backup':
            if has_key:
                print("[omp_num_threads] backup os.environ['%s'] = '%s'" %(key, os.environ[key]))
                omp_dct['num_threads'] = os.environ[key]
            else:
                omp_dct['num_threads'] = None
        elif action == 'restore':
            if has_key:
                print("[omp_num_threads] restoring os.environ['%s'] = '%s'" \
                    %(key, omp_dct['num_threads']))
                os.environ[key] = omp_dct['num_threads']
        elif action == 'set':
            print("[omp_num_threads] setting os.environ['%s'] = '%s'" %(key, str(num)))
            os.environ[key] = str(num)

    rand = np.random.rand

    nat = 10
    nstep = 500
    vel = rand(nstep, nat, 3)
    mass = rand(nat)
    cc = np.zeros((nstep,))

    bar = '-'*70

    ret = omp_num_threads('check', err=True)
    if ret == 'err':
        print(bar)
        print("""Do
        $ export OMP_NUM_THREADS=3
    before running this test.""")
        print(bar + '\n')
        sys.exit(0)

    omp_num_threads('check')
    omp_num_threads('backup')
    omp_num_threads('set', num=4)

    #-----------------------------------------------------------------------------

    print(bar)
    print("""testing _flib.vacf(vel,mass,cc,1,1), no nthreads from Python -- extension
    called directly ... does NOT read os.environ, reacts only if OMP_NUM_THREADS
    has been set in the shell BEFORE this test script was called""")
    omp_num_threads('check')
    cc = _flib.vacf(vel,mass,cc,1,1)
    print(bar + '\n')

    #-----------------------------------------------------------------------------

    print(bar)
    nthreads = 2
    print("testing _flib.vacf(vel,mass,cc,1,1,nthreads), setting nthreads = %i" %nthreads)
    cc = _flib.vacf(vel,mass,cc,1,1,nthreads)
    print(bar + '\n')

    #-----------------------------------------------------------------------------

    print(bar)
    print("testing _flib.vacf(vel,mass,cc,1,1), no nthreads from Python, take two")
    print("*" * 70)
    print("""!!! POSSIBLE F2PY BUG !!!
    After calling omp_set_num_threads() in the last test, OMP_NUM_THREADS is no
    longer recognized on the Fortran side!!! nthreads is still at the value from
    the last test: %s, that is WRONG
    !!! POSSIBLE F2PY BUG !!!""" %nthreads)
    print("*" * 70)
    omp_num_threads('check')
    cc = _flib.vacf(vel,mass,cc,1,1)
    print(bar + '\n')

    #-----------------------------------------------------------------------------

    print(bar)
    nthr = 2
    print("""testing pydos.fvacf(vel, m=mass, nthreads=%i) --
    override any OMP_NUM_THREADS setting in the environment AND os.environ""" %nthreads)
    cc = fvacf(vel, m=mass, nthreads=nthreads)
    print(bar + '\n')

    #-----------------------------------------------------------------------------

    print(bar)
    print("""testing pydos.fvacf(vel, m=mass, nthreads=None): no nthreads from Python -- It
    reads os.environ (workaround for f2py bug).""")
    omp_num_threads('check')
    cc = fvacf(vel, m=mass, nthreads=None)
    print(bar + '\n')

    omp_num_threads('restore')
