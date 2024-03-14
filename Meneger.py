import Preparation
import Test

Mode = 0
if (Mode == 0):
    N_train = 500000
    N_test = 50000
    l, d, l_test, d_test = Preparation.Open(N_train, N_test)
    Batch = 50
    Epoch = 10
    Preparation.Preparete(l, d, l_test, d_test, Batch, Epoch)
else:
    N_train = 600000
    N_test = 60000
    y_train, X_train, y_test, X_test = Preparation.Open(N_train, N_test)
    Test.Test(X_test, y_test)
    x=1