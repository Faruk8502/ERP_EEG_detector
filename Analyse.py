import h5py
from scipy.fftpack import fft, ifft
import numpy as np
import matplotlib.pyplot as plt

def Spectrum(x, Fs, N):
    f_max = Fs / 2
    T = 1/Fs
    t_max = T * N
    freq = np.arange(N) * Fs / N
    y = fft(x)
    # plt.plot(freq[1:int(N/2)], np.abs(y[1:int(N/2)]))
    # plt.show()
    P = np.abs(y[1: int(N/2) + 1])
    PSD = (P*P)/np.abs(y[0])
    return PSD

# with h5py.File('GIB-UVA ERP-BCI.hdf5', 'r') as f:
#     data = f['features']
#     labels = f['erp_labels']
#     l = labels[0:10000]
#     d = np.mean(data[2, 0:128, 1:9], 1)
# Fs = 128
# N = len(d)
# Spectrum(d, Fs, N)