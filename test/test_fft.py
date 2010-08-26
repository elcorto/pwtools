import numpy as np
from pwtools import signal
from scipy.fftpack import fft

y = np.random.rand(1000)
pwtools_ffty = signal.dft(y)
scipy_ffty = fft(y)
np.testing.assert_almost_equal(scipy_ffty, pwtools_ffty)
np.testing.assert_almost_equal(scipy_ffty.real, pwtools_ffty.real)
np.testing.assert_almost_equal(scipy_ffty.imag, pwtools_ffty.imag)
