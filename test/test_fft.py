import numpy as np
from math import pi, sqrt
from matplotlib import pyplot as plt
import pydos
from scipy.fftpack import fft
##fft = pydos.dft

y = np.random.rand(1000)
print "pydos fft ..."
pydos_ffty = pydos.dft(y)
print "scipy fft ..."
scipy_ffty = fft(y)
print "assert ..."
np.testing.assert_almost_equal(scipy_ffty, pydos_ffty)
np.testing.assert_almost_equal(scipy_ffty.real, pydos_ffty.real)
np.testing.assert_almost_equal(scipy_ffty.imag, pydos_ffty.imag)

print "sin() test ..."
print "==============\n"
print "c == <y(0)y(t)>"
# T up -> freq. resolution up ( == df down)
T = 10
# N up -> Nyquist freq. up
N = 1000 
t = np.linspace(0.0,T,N) 
dt = t[1]-t[0]
# sin(w*t) = sin(2*pi*f*t)
func = np.sin
freqs = [1.0, 2.0, 4.0]
y = np.zeros(len(t), dtype=float)
for fr in freqs:
    y += func(2.0*pi*fr*t)
print "freqs:", freqs

def nrm(a):
    """ vector norm """
    return sqrt(np.dot(a,a))

class Runner(object):
    def __init__(self):
        self.id = 1
        self.f1 = plt.figure()
        self.ax1 = self.f1.add_subplot(111)
        # span across monitor
        self.f2 = plt.figure(figsize=(23,6))
    
    def run(self, y, symb, label, title, dt=dt):
        print "\n"
        print title
        ffty = fft(y)
        N = len(ffty)
        # frequency axis in Hz, "f" in sin(2*pi*f*t)
        fy = np.fft.fftfreq(N, dt)

        # Nyquist freq. must be > highest freq. contained in the signal, otherwise
        # we get "folding" (see NR) and the fft is crap
        print "N: %s" %N
        print "Nyquist freq. %s Hz" %(1.0/(2*dt))
        print "freq. resolution [df]:      %e Hz" %(1.0/(N*dt))
##        print "freq. resolution [fftfreq]: %e Hz" %(fy[-1]-fy[-2])
        print "real/imag:", nrm(ffty.real) / nrm(ffty.imag)
        self.ax1.plot(fy, np.abs(ffty), symb, label=label)
        self.ax2 = self.f2.add_subplot('1' + '4' + str(self.id))
        self.ax2.plot(fy, np.abs(ffty.real), 'r', label=label + 'real')
        self.ax2.plot(fy, np.abs(ffty.imag), 'g', label=label + 'imag')
        self.ax2.set_title(title)
        self.ax2.legend()
        self.id += 1

runner = Runner()
#-------------------------------------------------------

runner.run(y, '-', 'fft(y)', "direct fft(y)")

#-------------------------------------------------------

title = "fft(yy), yy = y mirrored at t=0"
yy = np.concatenate((y[::-1],y[1:]))
runner.run(yy, '-', 'fft(yy)', title)

#-------------------------------------------------------

# VACF, for now don't mirror <y(0)y(t)> at t=0
title =  "fft(c), using c=pydos.acorr(v, method=6)"
v = y[1:]-y[:-1]
c = pydos.acorr(v, method=6)
##c = pydos.fvacf1d(y)
runner.run(c, '-', 'fft(c)', title)
plt.figure()
plt.plot(c)

#-------------------------------------------------------

# now, mirror
title = "fft(cc), cc = c mirrored at t=0"
cc = np.concatenate((c[::-1],c[1:]))
runner.run(cc, '-', 'fft(cc)', title)

#-------------------------------------------------------

runner.ax1.set_xlabel('f [Hz]')
runner.ax1.legend()
runner.ax2.set_xlabel('f [Hz]')
runner.f2.subplots_adjust(left=0.03, right=0.97) 

plt.show()
