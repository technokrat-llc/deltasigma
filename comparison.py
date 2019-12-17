import numpy as np
import pylab as plt
from numpy.fft import fft
from deltasigma import *
OSR = 32
H = synthesizeNTF(5, OSR, 1)
N = 8192 
fB = np.ceil(N/(2*OSR))
f = 85
u = 0.5*np.sin(2*np.pi*f/N*np.arange(N))
v = simulateDSM(u, H)[0]
plt.figure(figsize=(10, 7))
plt.subplot(2, 1, 1)
t = np.arange(85)
# the equivalent of MATLAB 'stairs' is step in matplotlib
plt.step(t, u[t], 'g', label='u(n)')
plt.hold(True)
plt.step(t, v[t], 'b', label='v(n)')
plt.axis([0, 85, -1.2, 1.2]);
plt.ylabel('u, v');
plt.xlabel('sample')
plt.legend()
plt.subplot(2, 1, 2)
spec = fft(v*ds_hann(N))/(N/4)
plt.plot(np.linspace(0, 0.5, N/2 + 1), dbv(spec[:N/2 + 1]))
plt.axis([0, 0.5, -120, 0])
plt.grid(True)
plt.ylabel('dBFS/NBW')
snr = calculateSNR(spec[:fB], f)
s = 'SNR = %4.1fdB' % snr
plt.text(0.25, -90, s)
s =  'NBW = %7.5f' % (1.5/N)
plt.text(0.25, -110, s)
plt.xlabel("frequency $1 \\\\rightarrow f_s$")