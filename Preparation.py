import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.preprocessing import OneHotEncoder
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from Train import CNN
from scipy.fftpack import fft, ifft
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

def Processing(N_train, N_test, features_train, features_test):
    win_size = 30
    interval = 1
    features_train_pd = np.zeros((N_train, 128, 8))
    features_test_pd =  np.zeros((N_test, 128, 8))
    # for i in range(0, N_train):
    #     print(i)
    #     for j in range(0, 8):
    #         features_train_pd[i, :, j] = Spectrogram(features_train[i, :, j], win_size, interval)

    for i in range(0, N_test):
        for j in range(0, 8):
            print(i)
            features_test_pd[i, :, j] = Spectrogram(features_test[i, :, j], win_size, interval)

    return features_train_pd, features_test_pd
def Spectrogram(x, win_size, interval):
    Fs = 128
    N = len(x)/2
    N_step = int(np.floor(N/interval))
    Spectrum_sum = np.zeros(128)
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

