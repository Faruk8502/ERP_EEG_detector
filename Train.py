import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import joblib
import Analyse

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
    l = labels[0:10000]
    d = np.mean(data[0:10000, 0:128, :], 2)
    t = target[4000:5000, 2]
    sq = sequences[0:1000]
    ri = r_i[0:1000]
    sb = subjects[1000:2000]
    tr = trials[701000:701615]

data_spect = np.zeros((10000, 64))
Fs = 128
N = np.shape(d)[1]
a = Analyse.Spectrum(d[0, :], Fs, N)
for i in range(0, 10000):
    data_spect[i, :] = Analyse.Spectrum(d[i, :], Fs, N)
# Разделение выборки на тренировочную и тестовую
X_train, X_test, y_train, y_test = train_test_split(data_spect, l, test_size=0.2)

# Создание модели многослойного перцептрона
model = MLPClassifier(hidden_layer_sizes=(100, 50, 25), activation='relu', solver='adam', max_iter=500)

# Обучение модели
model.fit(X_train, y_train)

# Сохранение модели
filename = 'model_params_2.pkl'
joblib.dump(model, filename)