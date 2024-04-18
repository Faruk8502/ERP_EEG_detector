import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score,  mean_squared_error, r2_score, roc_auc_score, roc_curve
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.utils import resample
from sklearn.preprocessing import OneHotEncoder
import os
import joblib
import pywt

def Test(X_train, y_train, X_test, y_test, trs_test, sbj_test):
    # Загружаем модель
    model = load_model('model_params_2.keras')
    # filename = 'model_params_3.pkl'
    # model = joblib.load(filename)

    # Определяем номер пациента
    subject_number = np.min(sbj_test) + 1
    # subject_trials = trs_test[sbj_test == subject_number]
    # trial_values = np.unique(subject_trials)
    index1 = int(len(y_test[sbj_test == subject_number]) * 0.6)
    index2 = len(y_test[sbj_test == subject_number])

    # Совершаем тонкую настройку модели
    model, X_subject_test, X_subject_test_pd, y_subject_test = Additional_training(model, subject_number, index1, index2, X_train, y_train, X_test, y_test, sbj_test)

    # Генерируем категориальный вектор меток, устанавливаем порог
    size = np.shape(X_subject_test_pd)[0]
    y_pred_0 = model.predict([X_subject_test, X_subject_test_pd], batch_size=1024)
    y_pred = np.zeros(size)
    for i in range(0, size):
        if(y_pred_0[i, 0] < 0.97):
            y_pred[i] = 1
        else:
            y_pred[i] = 0

    # Расчитываем показатели эффективности модели
    Metrics(y_subject_test, y_pred, y_pred_0)
    # for i in range(0, 10000):
    #     x[i, :] = Analyse.Spectrum(d[i, :], Fs, N)

def Additional_training(model, subject_number, index1, index2, X_train, y_train, X_test, y_test, sbj_test):
    lenght = 5000
    X_resampled_train, y_resampled_train = resample(X_train, y_train, stratify=y_train, n_samples=lenght, replace=False)
    X_subject_train = X_test[sbj_test == subject_number][0:index1]
    y_subject_train = y_test[sbj_test == subject_number][0:index1]
    X_subject_test = X_test[sbj_test == subject_number][index1:index2]
    y_subject_test = y_test[sbj_test == subject_number][index1:index2]
    X_tuning = np.concatenate((X_resampled_train, X_subject_train), axis=0)
    y_tuning = np.concatenate((y_resampled_train, y_subject_train), axis=0)
    y_tuning = One_Hot_Encoder(y_tuning.reshape((int(np.shape(X_tuning)[0]), 1)))
    test_size = 0.2
    X_tuning_train, X_tuning_valid, y_tuning_train, y_tuning_valid = train_test_split(X_tuning, y_tuning, test_size=test_size)
    N_train = np.shape(X_tuning_train)[0]
    N_test = np.shape(X_subject_test)[0]
    N_valid = np.shape(X_tuning_valid)[0]
    X_tuning_train_pd, X_subject_test_pd, X_tuning_valid_pd = Processing(N_train, N_test, N_valid, X_tuning_train, X_subject_test, X_tuning_valid)
    print("Конечный размер данных для тонкой настройки:", np.shape(y_tuning)[0])
    Batch = 1024
    Epoch = 500
    learning_rate = 0.001

    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001,
                                                mode='min', patience=10, verbose=1,
                                                restore_best_weights=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9,
                                         beta_2=0.999, amsgrad=False)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit([X_tuning_train, X_tuning_train_pd], y_tuning_train, batch_size=Batch,
              epochs=Epoch, validation_data=([X_tuning_valid, X_tuning_valid_pd], y_tuning_valid), callbacks=[callback])
    return model, X_subject_test, X_subject_test_pd, y_subject_test
def One_Hot_Encoder(y):
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(y)
    y_encodered = enc.transform(y).toarray()
    return y_encodered

def Processing(N_train, N_test, N_valid, X_train, X_test, X_valid):
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

    return features_train_wt, features_test_wt, features_valid_wt
def Wavelet(x):
    wavelet = 'morl'
    scales = np.arange(1, 10)
    coefficients, freq = pywt.cwt(x, scales, wavelet)
    return coefficients[8, 0:128]

def Metrics(y_true, y_pred, y_pred_0):
    TP = np.sum(y_pred[y_true==1])
    TN = len(y_true[y_true==0]) - np.sum(y_pred[y_true==0])
    FP = np.sum(y_pred[y_true==0])
    FN = np.sum(y_true[y_pred==0])
    sensetivity = TP / (TP + FN) * 100
    specificity = TN / (FP + TN) * 100
    accuracy = (TP + TN) / len(y_true) * 100
    print('ones:', np.sum(y_true))
    print('zeros:', len(y_true) - np.sum(y_true))
    print('True positive:', TP)
    print('False positive:', FP)
    print('True negative:', TN)
    print('False negative:', FN)
    print('sensetivity:', sensetivity)
    print('specificity:', specificity)
    print('accuracy:', accuracy)
    fpr, tpr, _ = roc_curve(y_true, 1 - y_pred_0[:, 0])

    # create ROC curve
    plt.plot(fpr, tpr)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

