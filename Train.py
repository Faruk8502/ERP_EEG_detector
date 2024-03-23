import Preparation
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.constraints import max_norm
import joblib
import Analyse
import Test
import os

def Neuronet_0(d, l, Fs):
    data_spect = np.zeros((10000, 64))
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

def CNN(X_train, X_valid, y_train, y_valid, X_test, y_test, Fs, Batch, Epoch):

    # Архитектура СНС
    input_reports = 128
    scales_reports = [64, 32, 16]
    activation = 'elu'
    dropout_rate = 0.25
    n_channels = 8
    n_filters = 8
    b1_units = list()
    learning_rate = 0.001

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
    # C1 = layers.Conv2D(8, (64, 1), kernel_initializer='he_normal', activation='elu', padding='same')(input)
    # C2 = layers.Conv2D(8, (32, 1), kernel_initializer='he_normal', activation='elu', padding='same')(input)
    # C3 = layers.Conv2D(8, (16, 1), kernel_initializer='he_normal', activation='elu', padding='same')(input)
    # D1 = layers.DepthwiseConv2D((1, 8), depth_multiplier=2, activation='elu', padding='valid')(C1)
    # D2 = layers.DepthwiseConv2D((1, 8), depth_multiplier=2, activation='elu', padding='valid')(C2)
    # D3 = layers.DepthwiseConv2D((1, 8), depth_multiplier=2, activation='elu', padding='valid')(C3)
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
    # C4 = layers.Conv2D(8, (16, 1), kernel_initializer='he_normal',  activation='elu', padding='same')(A1)
    # C5 = layers.Conv2D(8, (8, 1), kernel_initializer='he_normal',  activation='elu', padding='same')(A1)
    # C6 = layers.Conv2D(8, (4, 1), kernel_initializer='he_normal', activation='elu', padding='same')(A1)
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

    # model.summary()

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001,
                                                mode='min', patience=10, verbose=1,
                                                restore_best_weights=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9,
                                         beta_2=0.999, amsgrad=False)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, batch_size=Batch, epochs=Epoch, validation_data=(X_valid, y_valid), callbacks=[callback])

    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print("Test accuracy:", test_acc)
    # filename = 'model_params_3.pkl'
    # joblib.dump(model, filename)]
    model.save('my_model_exp.keras')
    # Test.Test(X_test, y_test)

