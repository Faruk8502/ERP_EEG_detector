import h5py
from scipy.fftpack import fft, ifft
import numpy as np
import matplotlib.pyplot as plt

def Spectrogram(x, win_size, interval):
    N = len(x)/2
    N_step = int(np.floor(N/interval))
    Spectrum_sum = np.zeros(int(N_step))
    for i in range(0, N_step):
        if(win_size + i*interval < N):
            _, PSD = Spectrum(x[0 + i*interval:win_size + i*interval], Fs, N)
            Spectrum_sum[i] = np.sum(PSD[5:25])
    return Spectrum_sum
def Spectrum(x, Fs, N):
    f_max = Fs / 2
    T = 1/Fs
    t_max = T * N
    freq = np.arange(N/2) * Fs / N
    y = fft(x)
    P = np.abs(y[1: int(N/2) + 1])
    PSD = (P*P)/np.abs(y[0])
    return freq, PSD


with h5py.File('GIB-UVA ERP-BCI.hdf5', 'r') as f:
    features = f['features']
    labels = f['erp_labels']
    subject = f['subjects']
    subject_exp = subject[:]
    patient = 0
    labels_exp = labels[subject_exp == patient]
    features_exp = features[subject_exp == patient]
    features_exp_1 = features_exp[labels_exp == 1]
    features_exp_1_mean = np.mean(features_exp_1, 0)
# Fs = 128
# N = len(features_exp_1_mean[:, 0])
# T = 1 / 128
# t_max = T * N
# t = np.arange(0, t_max, T)
#
# for i in range(0, 8):
#     win_size = 30
#     interval = 1
#     Spectrum_sum = Spectrogram(features_exp_1_mean[:, i], win_size, interval)
#     plt.plot(Spectrum_sum)
#     plt.show()

# for i in range(0, 8):
#     plt.plot(t, features_exp_1_mean[:, i])
#     plt.xlabel('с')
#     plt.ylabel('мкВ')
#     plt.show()
# plt.plot(t, features_exp_1[0, :, 0])
# plt.xlabel('с')
# plt.ylabel('мкВ')
# plt.show()
# for i in range(0, 8):
#     freq, PSD = Spectrum(features_exp_1_mean[:, i], Fs, N)
#     plt.plot(freq, PSD)
#     plt.xlabel('Гц')
#     plt.ylabel('PSD')
#     plt.show()