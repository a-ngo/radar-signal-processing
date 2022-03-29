import numpy as np 
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fft2, fftshift

Fs = 1000  # sampling frequency
T = 1 / Fs  # sampling period
L = 1500  # length of signal (1500 ms)
t = np.arange(0, L) * T  # time vector

# TODO: Form a signal containing a 77 Hz sinusoid of amplitude 0.7 
# and a 43Hz sinusoid of amplitude 2.
S = 0.7*np.sin(2*np.pi*50*t) + np.sin(2*np.pi*120*t)

# Corrupt the signal with noise
X = S + 2*np.random.normal(0,1,len(S))

# Plot the noise signal (X(t)) in time domain
# Note: It is difficult to identiy the frequency components by looking
# at the signal X(t)
plt.plot(X[1:50])
plt.title('Signal Corrupted with Zero-Mean Random Noise')
plt.xlabel('t (milliseconds)')
plt.ylabel('X(t)')
plt.show()

# TODO: Compute the Fourier transform of the signal
Y = fft(X)

# TODO : Compute the two-sided spectrum P2. 
# Then compute the single-sided spectrum P1 based on P2 
# and the even-valued signal length L.
P2 = abs(Y/L)
P1 = P2[0:int(L/2)]

f = Fs*(np.arange(0,L/2))/L;
fig, ax = plt.subplots(1)

ax.plot(f, P1)
plt.title('Single-Sided Amplitude Spectrum of X(t)')
plt.xlabel('f (Hz)')
plt.ylabel('|P1(f)|')