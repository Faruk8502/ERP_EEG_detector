import h5py
from scipy import signal
from scipy.fftpack import fft, ifft
import numpy as np
import matplotlib.pyplot as plt
import pywt

def Spectrogram(x, win_size, interval):
    Fs = 128
    t = np.arange(0, 1, 1/Fs)
    N = len(x)
    # Удаление тренда
    x = signal.detrend(x, axis=0, type='linear')
    plt.plot(x)
    plt.title('Исходный сигнал')

    # Домножение на оконную функцию
    window = signal.windows.hamming(N, sym=True)
    x = x * window
    # Фильтрация
    x_filtred = Filter(x)
    plt.figure()
    # plt.plot(x_filtred)
    # plt.title('Отфильтрованный сигнал')
    # plt.show()
    # Вейвлет преобразование
    Wevlet(x, t, N)
    #  Построение спектрограммы
    N_step = int(np.floor(N / interval))
    Spectrum_sum = np.zeros(int(N))
    for i in range(0, N_step):
        if(win_size + i*interval < N):
            freq, PSD = Spectrum(x[0 + i*interval:win_size + i*interval], Fs, win_size)
            # plt.plot(PSD[0:64])
            # plt.show()
            Spectrum_sum[i] = np.sum(np.abs(PSD[7:8])) + np.sum(np.abs(PSD[15:17]))
    return Spectrum_sum
def Spectrum(x, Fs, N):
    f_max = Fs/2
    T = 1/Fs
    t_max = T * N
    freq = np.arange(N/2) * Fs / N
    window = signal.windows.hamming(N, sym=True)
    x = x * window
    y = fft(x)
    # plt.plot(np.abs(y))
    # plt.show()
    A = np.zeros(int(N/2))
    A[0] = np.abs(y[0])/N
    A[1:int(N/2)] = 2*np.abs(y[1:int(N/2)])/N
    P = np.zeros(int(N / 2))
    P[0] = A[0]*A[0]
    P[1:int(N / 2)] = A[1:int(N / 2)]*A[1:int(N / 2)]/2
    PSD = (P)/(Fs / N)
    return freq, PSD
def Filter(x):
    # Задаем частоту дискретизации
    fs = 128  # частота дискретизации

    # Задаем параметры фильтра
    f1 = 3  # нижняя частота среза
    f2 = 15  # верхняя частота среза
    width = 5.0  # ширина полосы
    ripple_db = 20  # уровень затухания в децибелах

    # Генерируем полосовой цифровой фильтр
    N, beta = signal.kaiserord(ripple_db, width / (0.5 * fs))
    taps = signal.firwin(N, [f1, f2], width=width, window=('kaiser', beta), pass_zero=False, fs=fs)

    y = signal.filtfilt(taps, 1, x)
    #Отображаем частотную характеристику фильтра
    # w, h = signal.freqz(taps, worN=8000)
    # plt.figure()
    # plt.plot(0.5 * fs * w / np.pi, np.abs(h), 'b')
    # plt.axvline(f1, color='r', linestyle='--')
    # plt.axvline(f2, color='r', linestyle='--')
    # plt.xlim(0, 0.5 * fs)
    # plt.title("Frequency Response of Filter")
    # plt.xlabel('Frequency [Hz]')
    # plt.ylabel('Gain')
    # plt.grid()
    # plt.show()
    return y
def Wevlet(x, t, N):
    # _____\\\\\ Scipy \\\\\_____
    # scales = np.arange(1, 10)
    # Выбор вейвлета
    # wavelet = signal.ricker

    # Создание гребенки фильтров
    # coefficients = signal.cwt(x, wavelet, scales)

    # Построение результатов
    # plt.figure(figsize=(10, 6))
    # plt.imshow(np.abs(coefficients), extent=[0, 1, 1, 128], cmap='jet', aspect='auto', vmax=abs(coefficients).max(),
    #            vmin=-abs(coefficients).max())
    # plt.colorbar(label='Амплитуда')
    # plt.title('Результаты вейвлет-преобразования')
    # plt.xlabel('Время')
    # plt.ylabel('Масштаб')
    # plt.show()

    # Построение результатов
    # plt.figure(figsize=(10, 6))
    #
    # for i in range(len(scales)):
    #     plt.subplot(2, 5, i + 1)
    #     plt.plot(x, color='black', lw=0.5)
    #     plt.plot(coefficients[i], color='red', lw=0.5)
    #     plt.title('Масштаб {}'.format(scales[i]))
    #     plt.xticks([])
    #     plt.yticks([])
    #
    # plt.tight_layout()
    # plt.show()

    # _____\\\ PyWavelets \\\\\_____

    # Применение вейвлет-преобразования с вейвлетом Daubechies
    wavelet = 'morl'  # Выбор вейвлета Daubechies
    if((wavelet in pywt.wavelist(kind='discrete')) == True):
        level = 1
        coefficients_2 = pywt.wavedec(x, wavelet, level=level)

        # Визуализация результатов
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(t, x, label='ЭЭГ с волной P300')
        plt.xlabel('Время (с)')
        plt.ylabel('Амплитуда')
        plt.legend()

        plt.subplot(2, 1, 2)
        for i, coef in enumerate(coefficients_2[1:11]):
            plt.plot(coef, label=f'Уровень {i}')
        plt.xlabel('Индекс коэффициента')
        plt.ylabel('Амплитуда коэффициента')
        plt.legend()
        plt.title('Коэффициенты вейвлет-преобразования')
        plt.tight_layout()
        plt.show()

        maximum = max(map(len, coefficients_2))
        for every in coefficients_2:
            try:
                one = np.pad(some, maximum)
                two = np.pad(every, maximum)
                coeffs_array = np.concatenate((one, two), axis=1)
            except UnboundLocalError:
                some = every
            some = every

        # Построение результатов
        plt.imshow(coeffs_array, cmap='jet', aspect='auto')
        plt.colorbar()
        plt.show()
    else:
        scales = np.arange(1, 10)
        coefficients_2, freq = pywt.cwt(x, scales, wavelet)
        # Визуализация результатов
        plt.imshow(coefficients_2, extent=[0, 1, 1, 128], cmap='jet', aspect='auto', vmax=abs(coefficients_2).max(),
                   vmin=-abs(coefficients_2).max())
        plt.colorbar(label='Амплитуда коэффициентов')
        plt.xlabel('Время')
        plt.ylabel('Масштаб')
        plt.show()

        plt.figure(figsize=(10, 6))

        for i in range(len(scales)):
            plt.subplot(2, 5, i + 1)
            plt.plot(x, color='black', lw=0.5)
            plt.plot(coefficients_2[i], color='red', lw=0.5)
            plt.title('Масштаб {}'.format(scales[i]))
            plt.xticks([])
            plt.yticks([])

        plt.tight_layout()
        plt.show()
    print(pywt.wavelist())

with h5py.File('GIB-UVA ERP-BCI.hdf5', 'r') as f:
    features = f['features']
    labels = f['erp_labels']
    subject = f['subjects']
    trials = f['trials']
    subject_exp = subject[:]
    trials_exp = trials[:]
    patient = 54
    labels_exp = labels[trials_exp == patient]
    features_exp = features[trials_exp == patient]
    features_exp_1 = features_exp[labels_exp == 1]
    features_exp_1_mean = np.mean(features_exp_1, 0)
Fs = 128
N = len(features_exp_1_mean[:, 0])
T = 1 / 128
t_max = T * N
t = np.arange(0, t_max, T)

# for i in range(0, 8):
#     win_size = 60
#     interval = 1
#     Spectrum_sum = Spectrogram(features_exp_1[0, :, i], win_size, interval)
#     plt.plot(Spectrum_sum)
#     plt.show()

# for i in range(0, 8):
#     y = Filter(features_exp_1_mean[:, i])
#     Wevlet(features_exp_1_mean[:, i], t, N)
#     plt.plot(t, y[:])
#     plt.xlabel('с')
#     plt.ylabel('мкВ')
#     plt.show()
# plt.plot(t, features_exp_1[0, :, 0])
# plt.xlabel('с')
# plt.ylabel('мкВ')
# plt.show()
# for i in range(0, 8):
#     freq, PSD = Spectrum(features_exp_1_mean[:, i], Fs, N)
#     plt.plot(freq[1:int(N/2)], PSD[1:int(N/2)])
#     plt.xlabel('Гц')
#     plt.ylabel('PSD')
#     plt.show()