import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.preprocessing import OneHotEncoder
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
        trs = trials[0:N_train]
    return l, d, l_test, d_test, trs

def Preparete(l, d, l_test, d_test, trials, Batch, Epoch, desired_ratio, test_size, n_trials):
    Fs = 128
    # Train.Neuronet_0(d, l, Fs)
    X_resampled_trial = np.zeros((1, 128, 8))
    y_resampled_trial = np.zeros(1)
    for i in range(0, n_trials):
        lenght = int(0.2*len(l[trials==i]))
        X_resampled_trial_one, y_resampled_trial_one = resample(d[trials==i], l[trials==i],
                                                                stratify=l[trials==i], n_samples=lenght, replace=False)
        if (i == 0):
            X_resampled_trial, y_resampled_trial = X_resampled_trial_one, y_resampled_trial_one
        else:
            X_resampled_trial = np.concatenate((X_resampled_trial, X_resampled_trial_one), axis=0)
            y_resampled_trial = np.concatenate((y_resampled_trial, y_resampled_trial_one), axis=0)

    # Балансировка истинных и ложных элементов
    X_resampled_3d, y_resampled = Balancing(d, l, desired_ratio)
    # Разбиение данных на обучающую и тестовую выборки
    X_train, X_valid, y_train, y_valid = train_test_split(X_resampled_trial, y_resampled_trial, test_size=test_size)
    X_test, y_test = d_test, l_test
    print("Конечный размер матрицы данных:", np.shape(X_train))
    y_train_encodered = One_Hot_Encoder(y_train.reshape((int(np.shape(X_train)[0]), 1)))
    CNN(X_train, X_valid, y_train_encodered, y_valid, X_test, y_test, Fs, Batch, Epoch)
def One_Hot_Encoder(y):
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(y)
    y_encodered = enc.transform(y)
    return y_encodered
def Balancing(d, l, desired_ratio):
    # Разделение выборки на истинные и ложные элементы
    true_samples = d[l == 1]
    false_samples = d[l == 0]

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
