import numpy as np


def kernel_ridge_regression(lambda_=0.1, gamma=1):
    raw_data = np.loadtxt('https://www.csie.ntu.edu.tw/~htlin/mooc/datasets/mltech/hw2_lssvm_all.dat')
    raw_X = raw_data[:, :-1]
    raw_Y = raw_data[:, [-1]]
    X_train = raw_X[:400, :]
    Y_train = raw_Y[:400, :]
    X_test = raw_X[400:, :]
    Y_test = raw_Y[400:, :]

    N = X_train.shape[0]

    K = np.array([
        [
            kernel(X_train[row_index, :], X_train[column_index, :], gamma)
            for column_index in range(N)
        ]
        for row_index in range(N)
    ])
    beta = np.dot(np.linalg.inv(lambda_ * np.identity(N) + K), Y_train)

    Y_train_hat = np.array([
        np.dot(beta.T, kernel_one_versus_many(X_train[row_index, :], X_train, gamma))
        for row_index in range(X_train.shape[0])
    ]).reshape((-1, 1))

    Y_test_hat = np.array([
        np.dot(beta.T, kernel_one_versus_many(X_test[row_index, :], X_train, gamma))
        for row_index in range(X_test.shape[0])
    ]).reshape((-1, 1))

    E_in = np.dot((Y_train - Y_train_hat).T, Y_train - Y_train_hat) / N
    E_out = np.dot((Y_test - Y_test_hat).T, Y_test - Y_test_hat) / X_test.shape[0]

    E_in_01 = np.average(np.sign(Y_train_hat) != Y_train)
    E_out_01 = np.average(np.sign(Y_test_hat) != Y_test)

    print(E_in, E_out, E_in_01, E_out_01)

    return E_in, E_out, E_in_01, E_out_01


def kernel(x1, x2, gamma=1):
    return np.exp(
        - gamma * np.dot((x1 - x2).T, x1 - x2)
    )


def kernel_one_versus_many(x, X, gamma=1):
    return np.exp(- gamma * np.sum(np.square(X - x), axis=1))


if __name__ == '__main__':
    for gamma in [32, 2, 0.125]:
        for lambda_ in [0.001, 1, 1000]:
            kernel_ridge_regression(lambda_, gamma)
