import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
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

def CNN(X_train, X_valid, y_train, y_valid,  X_test, y_test, Fs, Batch, Epoch):

    # Архитектура СНС
    input = tf.keras.Input(shape=(128, 8, 1))
    C1 = layers.Conv2D(8, (64, 1), activation='relu', padding='same')(input)
    C2 = layers.Conv2D(8, (32, 1), activation='relu', padding='same')(input)
    C3 = layers.Conv2D(8, (16, 1), activation='relu', padding='same')(input)
    D1 = layers.DepthwiseConv2D((1, 8), depth_multiplier=2, activation='relu', padding='valid')(C1)
    D2 = layers.DepthwiseConv2D((1, 8), depth_multiplier=2, activation='relu', padding='valid')(C2)
    D3 = layers.DepthwiseConv2D((1, 8), depth_multiplier=2, activation='relu', padding='valid')(C3)
    N1 = layers.Concatenate(axis=3)([D1, D2, D3])
    A1 = layers.AveragePooling2D(pool_size=(4, 1))(N1)
    C4 = layers.Conv2D(8, (16, 1), activation='relu', padding='same')(A1)
    C5 = layers.Conv2D(8, (8, 1), activation='relu', padding='same')(A1)
    C6 = layers.Conv2D(8, (4, 1), activation='relu', padding='same')(A1)
    N2 = layers.Concatenate(axis=3)([C4, C5, C6])
    A2 = layers.AveragePooling2D(pool_size=(2, 1))(N2)
    C7 = layers.Conv2D(12, (8, 1), activation='relu', padding='same')(A2)
    A3 = layers.AveragePooling2D(pool_size=(2, 1))(C7)
    C8 = layers.Conv2D(6, (4, 1), activation='relu', padding='same')(A3)
    A4 = layers.AveragePooling2D(pool_size=(2, 1))(C8)
    F = layers.Flatten()(A4)
    output = layers.Dense(2, activation='softmax')(F)
    model = tf.keras.Model(inputs=input, outputs=output)

    # model.summary()

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, batch_size=Batch, epochs=Epoch, validation_data=(X_valid, y_valid))

    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print("Test accuracy:", test_acc)
    # filename = 'model_params_3.pkl'
    # joblib.dump(model, filename)]
    model.save('my_model_batch_50_data_600k.keras')
    Test.Test(X_test, y_test)

