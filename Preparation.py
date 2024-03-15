import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
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

def Preparete(l, d, l_test, d_test, Batch, Epoch, desired_ratio):
    Fs = 128
    # Train.Neuronet_0(d, l, Fs)

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
    X_resampled_3d = np.concatenate((true_samples, downsampled_false_samples))
    y_resampled = np.concatenate((true_labels, false_labels))

    # Разбиение данных на обучающую и тестовую выборки
    X_train, X_valid, y_train, y_valid = train_test_split(X_resampled_3d, y_resampled, test_size=0.2)
    X_test, y_test = d_test, l_test
    print("Конечный размер матрицы данных:", np.shape(X_train))
    CNN(X_train, X_valid, y_train, y_valid, X_test, y_test, Fs, Batch, Epoch)

