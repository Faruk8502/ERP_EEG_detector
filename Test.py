import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score,  mean_squared_error, r2_score, roc_auc_score, roc_curve
from tensorflow.keras.models import load_model
import joblib
import Analyse

def Test(X_test, y_test):
    model = load_model('my_model_exp.keras')
    # filename = 'model_params_3.pkl'
    # model = joblib.load(filename)

    # x = np.zeros((10000, 64))
    size = np.shape(X_test)[0]
    Fs = 128
    # for i in range(0, 10000):
    #     x[i, :] = Analyse.Spectrum(d[i, :], Fs, N)
    y_pred_0 = model.predict(X_test, batch_size=1024)
    y_pred = np.zeros(size)
    for i in range(0, size):
        if(y_pred_0[i, 0] < 0.97):
            y_pred[i] = 1
        else:
            y_pred[i] = 0
    Metrics(y_test, y_pred, y_pred_0)
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

    # Оценка точности модели
    # accuracy = model.score(d, y)
    # precision = accuracy_score(y_test, y_pred)
    # recall = recall_score(y_test, y_pred)
    # specivity = recall_score(y_test, y_pred, pos_label=0)
    # mse = mean_squared_error(y_test, y_pred)
    # r2 = roc_auc_score(y_test, y_pred)

    # print("Точность модели: {:.2f}%".format(accuracy * 100))
    # print("Толчность модели: {:.2f}%".format(precision * 100))
    # print("Специфичность модели: {:.2f}%".format(specivity * 100))
    # print("чувствительность:", recall*100, "%")
    # print("MSE:", mse)
    # print("R-squared:", r2)
    # print(np.sum(np.abs(y_test - y_pred)))
    # print(np.sum(y_test))
    #
    # fpr, tpr, _ = roc_curve(y_test, 1 - y_pred_0[:, 0])
    #
    # # create ROC curve
    # plt.plot(fpr, tpr)
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')
    # plt.show()