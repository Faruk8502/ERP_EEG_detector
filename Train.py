import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import joblib
import Analyse
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

def CNN(data, pred, Fs):
    # model = tf.keras.Sequential()
    #
    # input = tf.keras.Input(shape=(128, 8, 1))
    # 1st layer: 3 convolutional filters (64*1, 32*1, 16*1)
    # model.add(layers.Conv2D(65, (64, 1), activation='relu', input_shape=(128, 8, 1))(input)
    # model.add(layers.Conv2D(33, (32, 1), activation='relu', padding='same'))(input)
    # model.add(layers.Conv2D(17, (16, 1), activation='relu', padding='same'))(input)

    # 2nd layer: 3 convolutional filters (1*8, 1*8, 1*8)
    # model.add(layers.Conv2D(8, (1, 8), activation='relu', padding='same'))
    # model.add(layers.Conv2D(4, (1, 8), activation='relu', padding='same'))
    # model.add(layers.Conv2D(8, (1, 8), activation='relu', padding='same'))

    # # 3rd layer: concatenation
    # model.add(layers.Concatenate())

    # 4th layer: average pooling (1*4)
    # model.add(layers.AveragePooling2D(pool_size=(4, 1)))

    # Flatten the tensor
    # model.add(layers.Flatten())

    # Add a dense layer
    # model.add(layers.Dense(2, activation='softmax'))

    input = tf.keras.Input(shape=(128, 8, 1))
    C1 = layers.Conv2D(1, (64, 1), activation='relu')(input)
    C2 = layers.Conv2D(1, (32, 1), activation='relu')(input)
    C3 = layers.Conv2D(1, (16, 1), activation='relu')(input)
    D1 = layers.Conv2D(1, (1, 8), activation='relu')(C1)
    D2 = layers.Conv2D(1, (1, 8), activation='relu')(C2)
    D3 = layers.Conv2D(1, (1, 8), activation='relu')(C3)
    N1 = layers.Concatenate(axis=1)([D1, D2, D3])
    A1 = layers.AveragePooling2D(pool_size=(4, 1))(N1)
    C4 = layers.Conv2D(1, (16, 1), activation='relu')(A1)
    C5 = layers.Conv2D(1, (8, 1), activation='relu')(A1)
    C6 = layers.Conv2D(1, (4, 1), activation='relu')(A1)
    # D4 = layers.Conv2D(1, (4, 1), activation='relu')(C4)
    # D5 = layers.Conv2D(1, (4, 1), activation='relu')(C5)
    # D6 = layers.Conv2D(1, (4, 1), activation='relu')(C6)
    N2 = layers.Concatenate(axis=1)([C4, C5, C6])
    A2 = layers.AveragePooling2D(pool_size=(2, 1))(N2)
    C7 = layers.Conv2D(1, (8, 1), activation='relu')(A2)
    A3 = layers.AveragePooling2D(pool_size=(2, 1))(C7)
    C8 = layers.Conv2D(1, (4, 1), activation='relu')(A3)
    A3 = layers.AveragePooling2D(pool_size=(2, 1))(C8)
    F = layers.Flatten()(A3)
    output = layers.Dense(2, activation='softmax')(F)
    model = tf.keras.Model(inputs=input, outputs=output)

    # model.summary()

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Train the model
    X_train, X_test, y_train, y_test = train_test_split(data, pred, test_size=0.2)
    model.fit(data, pred, epochs=30, validation_data=(X_test, y_test))

    # # Evaluate the model
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print("Test accuracy:", test_acc)
    # filename = 'model_params_3.pkl'
    # joblib.dump(model, filename)]
    model.save('my_model_100.keras')