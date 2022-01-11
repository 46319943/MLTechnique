import numpy as np


def knn(K):
    train_raw = np.loadtxt('https://www.csie.ntu.edu.tw/~htlin/mooc/datasets/mltech/hw4_nbor_train.dat')
    test_raw = np.loadtxt('https://www.csie.ntu.edu.tw/~htlin/mooc/datasets/mltech/hw4_nbor_test.dat')

    X_train = train_raw[:, :-1]
    Y_train = train_raw[:, [-1]]
    X_test = test_raw[:, :-1]
    Y_test = test_raw[:, [-1]]

    X = X_train
    Y = Y_train

    y_train_hat_list = []
    for x in X_train:
        distance = np.sum(np.square(X_train - x), axis=1)
        ind = np.argpartition(distance, K)[:K]
        y = np.sign(np.sum(Y_train[ind]))
        y_train_hat_list.append(y)

    y_train_hat = np.array(y_train_hat_list).reshape(-1, 1)
    Ein = np.average(y_train_hat != Y_train)

    y_test_hat_list = []
    for x in X_test:
        distance = np.sum(np.square(X_train - x), axis=1)
        ind = np.argpartition(distance, K)[:K]
        y = np.sign(np.sum(Y_train[ind]))
        y_test_hat_list.append(y)

    y_test_hat = np.array(y_test_hat_list).reshape(-1, 1)
    Eout = np.average(y_test_hat != Y_test)

    print()

if __name__ == '__main__':
    knn(1)
    knn(5)
