import numpy as np 
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fft2, fftshift

Fs = 1000  # sampling frequency
T = 1 / Fs  # sampling period
L = 1500  # length of signal (1500 ms)
t = np.arange(0, L) * T  # time vector

# Form a signal containing a 77 Hz sinusoid of amplitude 0.7 
# and a 43Hz sinusoid of amplitude 2.
S = 0.7*np.sin(2*np.pi*50*t) + np.sin(2*np.pi*120*t)

# Corrupt the signal with noise
X = S + np.random.normal(0,1,len(S))

# Convert the signal in MxN matrix, 
# where M is the size of Range FFT samples and 
# N is the size of Doppler FFT samples

# let
M = int(len(X) / 50)
N = int(len(X) / 30)

X_2d = np.reshape(X, (M,N))
plt.matshow(X_2d)
plt.colorbar()
plt.show()

# TODO: Compute the 2D Fourier Transform of the data
Y_2d = fft2(X_2d)

# TODO: Shift the zero-frequency component to the center of the 
# output and plot the resulting matrix, which is the same size as X_2d.
plt.matshow(abs(fftshift(Y_2d)))
plt.colorbar()
plt.show()