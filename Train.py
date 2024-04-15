import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.constraints import max_norm
import joblib
# import Analyse

def Neuronet_0(d, l, Fs):
    # data_spect = np.zeros((10000, 64))
    # N = np.shape(d)[1]
    # a = Analyse.Spectrum(d[0, :], Fs, N)
    # for i in range(0, 10000):
    #     data_spect[i, :] = Analyse.Spectrum(d[i, :], Fs, N)
    # Разделение выборки на тренировочную и тестовую
    X_train, X_test, y_train, y_test = train_test_split(d, l, test_size=0.2)

    # Создание модели многослойного перцептрона
    model = MLPClassifier(hidden_layer_sizes=(100, 50, 25), activation='relu', solver='adam', max_iter=500)

    return model

def CNN(input_reports, scales_reports, activation, dropout_rate, n_channels, n_filters):

    # Архитектура СНС
    b1_units = list()

    # Первый блок_____________________________________________
    input = tf.keras.Input(shape=(input_reports, n_channels, 1))
    for i in range(len(scales_reports)):
        unit = layers.Conv2D(n_filters,
                             kernel_size=(scales_reports[i], 1),
                             kernel_initializer='he_normal',
                             padding='same')(input)
        unit = layers.BatchNormalization()(unit)
        unit = layers.Activation(activation)(unit)
        unit = layers.Dropout(dropout_rate)(unit)
        unit = layers.DepthwiseConv2D((1, n_channels),
                                      use_bias=False,
                                      depth_multiplier=2,
                                      depthwise_constraint=max_norm(1.))(unit)
        unit = layers.BatchNormalization()(unit)
        unit = layers.Activation(activation)(unit)
        unit = layers.Dropout(dropout_rate)(unit)
        b1_units.append(unit)
    N1 = layers.Concatenate(axis=3)(b1_units)
    A1 = layers.AveragePooling2D(pool_size=(4, 1))(N1)

    # Второй блок_____________________________________________
    b2_units = list()
    for i in range(len(scales_reports)):
        unit = layers.Conv2D(n_filters,
                             kernel_size=(int(scales_reports[i]/4), 1),
                             kernel_initializer='he_normal',
                             use_bias=False,
                             padding='same')(A1)
        unit = layers.BatchNormalization()(unit)
        unit = layers.Activation(activation)(unit)
        unit = layers.Dropout(dropout_rate)(unit)
        b2_units.append(unit)
    N2 = layers.Concatenate(axis=3)(b2_units)
    A2 = layers.AveragePooling2D(pool_size=(2, 1))(N2)

    # Третий блок_____________________________________________
    C7 = layers.Conv2D(int(n_filters*len(scales_reports)/2),
                       kernel_size=(8, 1),
                       kernel_initializer='he_normal',
                       use_bias=False,
                       padding='same')(A2)
    C7_1 = layers.BatchNormalization()(C7)
    C7_2 = layers.Activation(activation)(C7_1)
    C7_3 = layers.AveragePooling2D(pool_size=(2, 1))(C7_2)
    C7_4 = layers.Dropout(dropout_rate)(C7_3)

    C8 = layers.Conv2D(int(n_filters*len(scales_reports)/2),
                       kernel_size=(4, 1),
                       kernel_initializer='he_normal',
                       use_bias=False,
                       padding='same')(C7_4)
    C8_1 = layers.BatchNormalization()(C8)
    C8_2 = layers.Activation(activation)(C8_1)
    C8_3 = layers.AveragePooling2D((2, 1))(C8_2)
    C8_4 = layers.Dropout(dropout_rate)(C8_3)

    F = layers.Flatten()(C8_4)
    output = layers.Dense(2, activation='softmax')(F)
    model = tf.keras.Model(inputs=input, outputs=output)
    return model

def CNN_2(input_reports, scales_reports, activation, dropout_rate, n_channels, n_filters):

    # Архитектура СНС

    input = tf.keras.Input(shape=(input_reports, n_channels, 1))
    C1 = layers.Conv2D(n_filters,
                         kernel_size=(32, 1),
                         kernel_initializer='he_normal',
                         padding='same')(input)
    C1_BN = layers.BatchNormalization()(C1)
    C1_Act = layers.Activation(activation)(C1_BN)
    C1_Dout = layers.Dropout(dropout_rate)(C1_Act)
    D1 = layers.DepthwiseConv2D((1, n_channels),
                                  use_bias=False,
                                  depth_multiplier=2,
                                  depthwise_constraint=max_norm(1.))(C1_Dout)
    D1_BN = layers.BatchNormalization()(D1)
    D1_Act = layers.Activation(activation)(D1_BN)
    D1_Dout = layers.Dropout(dropout_rate)(D1_Act)
    A1 = layers.AveragePooling2D(pool_size=(4, 1))(D1_Dout)
    C2 = layers.Conv2D(n_filters,
                       kernel_size=(8, 1),
                       kernel_initializer='he_normal',
                       padding='same')(A1)
    C2_BN = layers.BatchNormalization()(C2)
    C2_Act = layers.Activation(activation)(C2_BN)
    C2_Dout = layers.Dropout(dropout_rate)(C2_Act)
    A2 = layers.AveragePooling2D(pool_size=(2, 1))(C2_Dout)
    F = layers.Flatten()(A2)
    output = layers.Dense(2, activation='softmax')(F)
    model = tf.keras.Model(inputs=input, outputs=output)
    return model
