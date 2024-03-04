import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score,  mean_squared_error, r2_score, roc_auc_score
import joblib
import Analyse


filename = 'model_params_2.pkl'
model = joblib.load(filename)

with h5py.File('GIB-UVA ERP-BCI.hdf5', 'r') as f:
    data = f['features']
    labels = f['erp_labels']
    d = np.mean(data[10000:20000, 0:128, :], 2)
    y = labels[10000:20000]
x = np.zeros((10000, 64))
Fs = 128
N = np.shape(d)[1]
for i in range(0, 10000):
    x[i, :] = Analyse.Spectrum(d[i, :], Fs, N)
y_pred = model.predict(x)
# Оценка точности модели
accuracy = model.score(x, y)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
mse = mean_squared_error(y, y_pred)
r2 = roc_auc_score(y, y_pred)

print("Точность модели: {:.2f}%".format(accuracy * 100))
print("Чувствительность модели: {:.2f}%".format(precision * 100))
print("Специфичность модели: {:.2f}%".format(accuracy * 100))
print("MSE:", mse)
print("R-squared:", r2)
print(np.sum(np.abs(y - y_pred)))