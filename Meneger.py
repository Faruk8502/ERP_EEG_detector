import Preparation
import Test
import numpy as np

Mode = 0
if (Mode == 0):
    N_train = 500000
    N_test = 50000
    l, d, l_test, d_test, trials = Preparation.Open(N_train, N_test)
    Batch = 1024
    Epoch = 20
    desired_ratio = 1
    test_size = 0.2
    n_trials = np.max(trials)
    Preparation.Preparete(l, d, l_test, d_test, trials, Batch, Epoch, desired_ratio, test_size, n_trials)
else:
    N_train = 600000
    N_test = 100000
    y_train, X_train, y_test, X_test = Preparation.Open(N_train, N_test)
    Test.Test(X_test, y_test)
    x=1