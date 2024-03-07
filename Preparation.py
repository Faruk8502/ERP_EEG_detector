import h5py
import numpy as np
import Train
from sklearn.model_selection import train_test_split
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler

with h5py.File('GIB-UVA ERP-BCI.hdf5', 'r') as f:
    for key in f.keys():
        print(key)
    data = f['features']
    labels = f['erp_labels']
    target = f['target']
    m_d = f['matrix_dims']
    r_i = f['run_indexes']
    sequences = f['sequences']
    subjects = f['subjects']
    trials = f['trials']
    print(target)
    print(labels)
    print(data)
    print(m_d)
    print(sequences)
    print(r_i)
    print(subjects)
    print(trials)
    l = labels[0:250000]
    d = data[0:250000, 0:128, 0:8]
    t = target[4000:5000, 2]
    sq = sequences[0:1000]
    ri = r_i[0:1000]
    sb = subjects[1000:2000]
    tr = trials[701000:701615]
Fs = 128
# Train.Neuronet_0(d, l, Fs)
# Преобразование в двумерную форму данных (n_samples, n_features)
X_flat = d.reshape(d.shape[0], -1)

# Применение RandomUnderSampler
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X_flat, l)

# Восстановление трехмерной формы данных
X_resampled_3d = X_resampled.reshape(X_resampled.shape[0], 128, 8)
Train.CNN(X_resampled_3d, y_resampled, Fs)