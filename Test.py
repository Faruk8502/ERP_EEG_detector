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
import Analyse

def Test(X_train, y_train, X_test, y_test, trs_test, sbj_test):
    # Загружаем модель
    model = load_model('model_params_3.keras')
    # filename = 'model_params_3.pkl'
    # model = joblib.load(filename)

    # Определяем номер пациента
    subject_number = np.min(sbj_test) + 1
    # subject_trials = trs_test[sbj_test == subject_number]
    # trial_values = np.unique(subject_trials)
    start_index = int(len(y_test[sbj_test == subject_number]) * 0.6)
    end_index = len(y_test[sbj_test == subject_number])

    # Совершаем тонкую настройку модели
    model = Additional_training(model, subject_number, start_index, X_train, y_train, X_test, y_test, sbj_test)

    # Генерируем категориальный вектор меток, устанавливаем порог
    size = np.shape(X_test[sbj_test == subject_number][start_index:end_index])[0]
    y_pred_0 = model.predict(X_test[sbj_test == subject_number][start_index:end_index], batch_size=1000)
    y_pred = np.zeros(size)
    for i in range(0, size):
        if(y_pred_0[i, 0] < 0.97):
            y_pred[i] = 1
        else:
            y_pred[i] = 0

    # Расчитываем показатели эффективности модели
    Metrics(y_test[sbj_test == subject_number][start_index:end_index], y_pred, y_pred_0)
    # for i in range(0, 10000):
    #     x[i, :] = Analyse.Spectrum(d[i, :], Fs, N)

def Additional_training(model, subject_number, end_index, X_train, y_train, X_test, y_test, sbj_test):
    lenght = 5000
    X_resampled_train, y_resampled_train = resample(X_train, y_train, stratify=y_train, n_samples=lenght, replace=False)
    X_subject_train = X_test[sbj_test == subject_number][0:end_index]
    y_subject_train = y_test[sbj_test == subject_number][0:end_index]
    X_tuning = np.concatenate((X_resampled_train, X_subject_train), axis=0)
    y_tuning = np.concatenate((y_resampled_train, y_subject_train), axis=0)
    y_tuning = One_Hot_Encoder(y_tuning.reshape((int(np.shape(X_tuning)[0]), 1)))
    test_size = 0.2
    X_tuning_train, X_tuning_valid, y_tuning_train, y_tuning_valid = train_test_split(X_tuning, y_tuning, test_size=test_size)
    print("Конечный размер данных для тонкой настройки:", np.shape(y_tuning)[0])
    Batch = 512
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
    model.fit(X_tuning, y_tuning, batch_size=Batch, epochs=Epoch, validation_data=(X_tuning_valid, y_tuning_valid), callbacks=[callback])
    return model
def One_Hot_Encoder(y):
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(y)
    y_encodered = enc.transform(y).toarray()
    return y_encodered
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

