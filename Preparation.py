import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.preprocessing import OneHotEncoder
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from Train import CNN
from scipy.fftpack import fft, ifft
from scipy import signal
import pywt
def Open(N_train, N_test):
    with h5py.File('GIB-UVA ERP-BCI.hdf5', 'r') as f:
        # for key in f.keys():
        #     print(key)
        features = f['features']
        labels = f['erp_labels']
        target = f['target']
        m_d = f['matrix_dims']
        r_i = f['run_indexes']
        sequences = f['sequences']
        subjects = f['subjects']
        trials = f['trials']
        labels_train = labels[0:N_train]
        features_train = features[0:N_train, 0:128, 0:8]
        features_test = features[N_train:N_train + N_test, 0:128, 0:8]
        labels_test = labels[N_train:N_train + N_test]
        trs_train = trials[0:N_train]
        trs_test = trials[N_train:N_train + N_test]
        sbj_test = subjects[N_train:N_train + N_test]
    return labels_train, features_train, labels_test, features_test, trs_train, trs_test, sbj_test

def Preparete(labels_train, features_train, labels_test, features_test, trs_train, n_trials, desired_ratio, test_size):
    # Балансировка испытаний
    X_trial_balanced, y_trial_balanced = Trial_Balancing(labels_train, features_train, n_trials, trs_train)

    # Балансировка истинных и ложных элементов
    X_resampled, y_resampled = Balancing(labels_train, features_train, desired_ratio)

    # Разбиение данных на обучающую и тестовую выборки
    X_train, X_valid, y_train, y_valid = train_test_split(features_train, labels_train, test_size=test_size)
    X_test, y_test = features_test, labels_test
    print("Конечный размер матрицы обучающих данных:", np.shape(X_train))

    # One-hot кодирование векторов с метками
    y_train_encodered = One_Hot_Encoder(y_train.reshape((int(np.shape(X_train)[0]), 1)))
    y_valid_encodered = One_Hot_Encoder(y_valid.reshape((int(np.shape(X_valid)[0]), 1)))
    y_test_encodered = One_Hot_Encoder(y_test.reshape((int(np.shape(X_test)[0]), 1)))

    return X_train, y_train_encodered, X_valid, y_valid_encodered, X_test, y_test_encodered
def One_Hot_Encoder(y):
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(y)
    y_encodered = enc.transform(y).toarray()
    return y_encodered

def Trial_Balancing(labels, features, n_trials, trials):
    X_resampled_trial = np.zeros((1, 128, 8))
    y_resampled_trial = np.zeros(1)
    for i in range(0, n_trials):
        lenght = int(0.5*len(labels[trials == i]))
        X_resampled_trial_one, y_resampled_trial_one = \
            resample(features[trials == i], labels[trials == i],
                     stratify=labels[trials == i], n_samples=lenght, replace=False)
        if (i == 0):
            X_resampled_trial, y_resampled_trial = X_resampled_trial_one, y_resampled_trial_one
        else:
            X_resampled_trial = np.concatenate((X_resampled_trial, X_resampled_trial_one), axis=0)
            y_resampled_trial = np.concatenate((y_resampled_trial, y_resampled_trial_one), axis=0)
    return X_resampled_trial, y_resampled_trial
def Balancing(labels, features, desired_ratio):
    # Разделение выборки на истинные и ложные элементы
    true_samples = features[labels == 1]
    false_samples = features[labels == 0]

    # Получение количества истинных элементов, соответствующего заданному соотношению
    n_true_samples = np.shape(true_samples)[0]
    n_false_samples = int(np.ceil(n_true_samples * desired_ratio))

    # Получение соотвтествующих меток
    true_labels = np.ones(n_true_samples)
    false_labels = np.zeros(n_false_samples)

    # Отбрасываем случайным образом избыточные истинные элементы
    downsampled_false_samples = resample(false_samples, n_samples=n_false_samples, replace=False)

    # Объединение истинных и ложных элементов для формирования новой выборки
    X_balanced = np.concatenate((true_samples, downsampled_false_samples))
    y_balanced = np.concatenate((true_labels, false_labels))
    return X_balanced, y_balanced
#_______________________________________________________________________________________________________________________

def Processing(N_train, N_test, N_valid, X_train, X_test, X_valid):
    # win_size = 24
    # interval = 1
    # features_train_pd = np.zeros((N_train, 128, 8))
    # features_test_pd =  np.zeros((N_test, 128, 8))
    features_train_wt = np.zeros((N_train, 128, 8))
    features_test_wt = np.zeros((N_test, 128, 8))
    features_valid_wt = np.zeros((N_valid, 128, 8))
    for i in range(0, N_train):
        print(i)
        for j in range(0, 8):
            features_train_wt[i, 0:128, j] = Wavelet(X_train[i, 0:128, j])
    for i in range(0, N_test):
        print(i)
        for j in range(0, 8):
            features_test_wt[i, 0:128, j] = Wavelet(X_test[i, 0:128, j])
    for i in range(0, N_valid):
        print(i)
        for j in range(0, 8):
            features_valid_wt[i, 0:128, j] = Wavelet(X_valid[i, 0:128, j])
    # for i in range(0, N_train):
    #     print(i)
    #     for j in range(0, 8):
    #         features_train_pd[i, 0:128, j] = Spectrogram(features_train[i, :, j], win_size, interval)
    #
    # for i in range(0, N_test):
    #     print(i)
    #     for j in range(0, 8):
    #         features_test_pd[i, 0:128, j] = Spectrogram(features_test[i, :, j], win_size, interval)

    return features_train_wt, features_test_wt, features_valid_wt
def Spectrogram(x, win_size, interval):
    Fs = 128
    N = len(x)
    N_step = int(np.floor(N/interval))
    Spectrum_sum = np.zeros(int(N))
    x_filtred = Filter(x)
    # for i in range(0, N_step):
    #     if(win_size + i*interval < N):
    #         _, PSD = Spectrum(x_filtred[0 + i*interval:win_size + i*interval], Fs, win_size)
    #         Spectrum_sum[i] = np.sum(PSD[3:5])
    return x_filtred

def Filter(x):
    # Задаем частоту дискретизации
    fs = 128  # частота дискретизации

    # Задаем параметры фильтра
    f1 = 2  # нижняя частота среза
    f2 = 5  # верхняя частота среза
    width = 3.0  # ширина полосы
    ripple_db = 20  # уровень затухания в децибелах

    # Генерируем полосовой цифровой фильтр
    N, beta = signal.kaiserord(ripple_db, width / (0.5 * fs))
    taps = signal.firwin(N, [f1, f2], width=width, window=('kaiser', beta), pass_zero=False, fs=fs)

    y = signal.filtfilt(taps, 1, x)
    return y
def Spectrum(x, Fs, N):
    f_max = Fs/2
    T = 1/Fs
    t_max = T * N
    freq = np.arange(N/2) * Fs / N
    y = fft(x)
    A = np.zeros(int(N/2))
    A[0] = np.abs(y[0])/N
    A[1:int(N/2)] = 2*np.abs(y[1:int(N/2)])/N
    P = np.zeros(int(N / 2))
    P[0] = A[0]*A[0]
    P[1:int(N / 2)] = A[1:int(N / 2)]*A[1:int(N / 2)]/2
    PSD = (P)/(Fs / N)
    return freq, PSD

def Wavelet(x):
    wavelet = 'morl'
    scales = np.arange(1, 10)
    coefficients, freq = pywt.cwt(x, scales, wavelet)
    return coefficients[8, 0:128]

