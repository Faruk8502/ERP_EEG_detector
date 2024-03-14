import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from Train import CNN
def Open(N_train, N_test):
    with h5py.File('GIB-UVA ERP-BCI.hdf5', 'r') as f:
        # for key in f.keys():
        #     print(key)
        data = f['features']
        labels = f['erp_labels']
        target = f['target']
        m_d = f['matrix_dims']
        r_i = f['run_indexes']
        sequences = f['sequences']
        subjects = f['subjects']
        trials = f['trials']
        l = labels[0:N_train]
        d = data[0:N_train, 0:128, 0:8]
        d_test = data[N_train:N_train + N_test, 0:128, 0:8]
        l_test = labels[N_train:N_train + N_test]
    return l, d, l_test, d_test

def Preparete(l, d, l_test, d_test, Batch, Epoch):
    Fs = 128
    # Train.Neuronet_0(d, l, Fs)
    # Преобразование в двумерную форму данных (n_samples, n_features)
    X_flat = d.reshape(d.shape[0], -1)

    # Применение RandomUnderSampler
    rus = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = rus.fit_resample(X_flat, l)

    # Восстановление трехмерной формы данных
    X_resampled_3d = X_resampled.reshape(X_resampled.shape[0], 128, 8)

    # Разбиение данных на обучающую и тестовую выборки
    X_train, X_valid, y_train, y_valid = train_test_split(X_resampled_3d, y_resampled, test_size=0.2)
    X_test, y_test = d_test, l_test
    print("Конечный размер матрицы данных:", np.shape(X_train))
    CNN(X_train, X_valid, y_train, y_valid, X_test, y_test, Fs, Batch, Epoch)

