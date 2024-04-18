import Preparation
import Test
import Train
import numpy as np
import tensorflow as tf
import joblib
import os

Mode = 1
if (Mode == 0):
    # _____ Определяем размеры обучающей и тестовой выборок_____
    N_train = 650000
    N_test = 50000
    #_____Извлекаем все необходимые данные из БД_____
    labels_train, features_train, labels_test, features_test,\
    trs_train, trs_test, sbj_test = Preparation.Open(N_train, N_test)
    # _____ Подготавливаем данные _____
    test_size = 0.2
    desired_ratio = 1  # отношение нулей и единиц
    n_trials = np.max(trs_train)
    X_train, y_train, X_valid, y_valid, X_test, y_test = \
        Preparation.Preparete(labels_train, features_train, labels_test,
                              features_test, trs_train, n_trials, desired_ratio, test_size)
    # _____ Строим модель _____
    input_reports = 128
    scales_reports = [64, 32, 16]
    activation = 'elu'
    dropout_rate = 0.25
    n_channels = 8
    n_filters = 8
    model = Train.CNN(input_reports, scales_reports, activation, dropout_rate, n_channels, n_filters)
    model.summary()
    # _____ Обучаем модель _____
    Epoch = 300
    Batch = 1024
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
    model.fit(X_train, y_train, batch_size=Batch, epochs=Epoch, validation_data=(X_valid, y_valid), callbacks=[callback])

    # _____ Тестируем модель _____
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print("Test accuracy:", test_acc)
    filename = 'model_params_3.keras'
    model.save(filename)
    Test.Test(X_train, y_train, X_test, y_test, trs_test, sbj_test)
elif (Mode == 1):
    # _____ Определяем размеры обучающей и тестовой выборок_____
    N_train = 300000
    N_test = 1000
    # _____Извлекаем все необходимые данные из БД_____
    labels_train, features_train, labels_test, features_test, \
    trs_train, trs_test, sbj_test = Preparation.Open(N_train, N_test)

    # _____ Подготавливаем данные _____
    test_size = 0.2
    desired_ratio = 1  # отношение нулей и единиц
    n_trials = np.max(trs_train)
    X_train, y_train, X_valid, y_valid, X_test, y_test = \
        Preparation.Preparete(labels_train, features_train, labels_test,
                              features_test, trs_train, n_trials, desired_ratio, test_size)

    # ____Обрабатываем данные_____
    N_valid = np.shape(X_valid)[0]
    N_train = np.shape(X_train)[0]
    X_train_pd, X_test_pd, X_valid_pd = \
        Preparation.Processing(N_train, N_test, N_valid, X_train, X_test, X_valid)

    # _____ Строим модель _____
    input_reports = 128
    scales_reports = [64, 32, 16]
    activation = 'elu'
    dropout_rate = 0.25
    n_channels = 8
    n_filters = 8
    model = Train.double_CNN(input_reports, scales_reports, activation, dropout_rate, n_channels, n_filters)
    model.summary()
    # _____ Обучаем модель _____
    Epoch = 300
    Batch = 1024
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
    model.fit([X_train, X_train_pd], y_train, batch_size=Batch, epochs=Epoch, validation_data=([X_valid, X_valid_pd], y_valid),
              callbacks=[callback])
    filename = 'model_params_2.keras'
    model.save(filename)
    # _____ Тестируем модель _____
    test_loss, test_acc = model.evaluate([X_test, X_test_pd], y_test, verbose=2)
    print("Test accuracy:", test_acc)
    Test.Test(X_train, y_train, X_test, y_test, trs_test, sbj_test)
elif (Mode == 2):
    # _____ Определяем размеры обучающей и тестовой выборок_____
    N_train = 100000
    N_test = 10
    # _____Извлекаем все необходимые данные из БД_____
    labels_train, features_train, labels_test, features_test, \
    trs_train, trs_test, sbj_test = Preparation.Open(N_train, N_test)

    # ____Обрабатываем данные_____
    features_train_pd, features_test_pd = \
        Preparation.Processing(N_train, N_test, features_train, features_test)
    # _____ Подготавливаем данные _____
    test_size = 0.2
    desired_ratio = 1  # отношение нулей и единиц
    n_trials = np.max(trs_train)
    X_train, y_train, X_valid, y_valid, X_test, y_test = \
        Preparation.Preparete(labels_train, features_train_pd, labels_test,
                              features_test_pd, trs_train, n_trials, desired_ratio, test_size)
    # _____ Обучаем модель _____
    model = Train.Neuronet_0(X_train, y_train, 128)
    # Обучение модели
    model.fit(X_train, y_train)

    # Сохранение модели
    filename = 'model_params_2.pkl'
    joblib.dump(model, filename)
elif (Mode == 3):
    N_train = 650000
    N_test = 11000
    y_train, X_train, y_test, X_test, trs_train, trs_test, sbj_test = Preparation.Open(N_train, N_test)
    Test.Test(X_train, y_train, X_test, y_test, trs_test, sbj_test)