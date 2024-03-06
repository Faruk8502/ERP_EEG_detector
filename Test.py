import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score,  mean_squared_error, r2_score, roc_auc_score
from tensorflow.keras.models import load_model
import joblib
import Analyse


# filename = 'model_params_3.pkl'
# model = joblib.load(filename)
model = load_model('my_model.keras')

with h5py.File('GIB-UVA ERP-BCI.hdf5', 'r') as f:
    data = f['features']
    labels = f['erp_labels']
    d = data[10000:20000, 0:128, 0:8]
    y = labels[10000:20000]
# x = np.zeros((10000, 64))
size = 10000
Fs = 128
N = np.shape(d)[1]
# for i in range(0, 10000):
#     x[i, :] = Analyse.Spectrum(d[i, :], Fs, N)
y_pred_0 = model.predict(d)
y_pred = np.zeros(size)
for i in range(0, size):
    if(y_pred_0[i, 0] > y_pred_0[i, 1]):
        y_pred[i] = 0
    else:
        y_pred[i] = 1
# Оценка точности модели
# accuracy = model.score(d, y)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
mse = mean_squared_error(y, y_pred)
r2 = roc_auc_score(y, y_pred)

# print("Точность модели: {:.2f}%".format(accuracy * 100))
print("Чувствительность модели: {:.2f}%".format(precision * 100))
# print("Специфичность модели: {:.2f}%".format(accuracy * 100))
print("MSE:", mse)
print("R-squared:", r2)
print(np.sum(np.abs(y - y_pred)))