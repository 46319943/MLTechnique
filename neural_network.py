import numpy as np


def backprop_stochastic(M, r, eta, T=50000):
    '''
    d - M - 1 neural network with tanh-type neurons, including the output neuron
    :return:
    '''

    train_raw = np.loadtxt('https://www.csie.ntu.edu.tw/~htlin/mooc/datasets/mltech/hw4_nnet_train.dat')
    test_raw = np.loadtxt('https://www.csie.ntu.edu.tw/~htlin/mooc/datasets/mltech/hw4_nnet_test.dat')

    X_train = train_raw[:, :-1]
    Y_train = train_raw[:, [-1]]
    X_test = test_raw[:, :-1]
    Y_test = test_raw[:, [-1]]

    X = X_train
    Y = Y_train

    N = X.shape[0]
    d = X.shape[1]

    W1 = (2 * np.random.random((d + 1, M)) - 1) * r
    W2 = (2 * np.random.random((M + 1, 1)) - 1) * r

    for i in range(T):
        random_index = np.random.choice(N, 1)
        X0 = X[random_index, :]
        Y_ = Y[random_index]

        X0 = np.concatenate([np.ones((X0.shape[0], 1)), X0], axis=1)
        S1 = np.dot(X0, W1)
        X1 = np.tanh(S1)

        X1 = np.concatenate([np.ones((X1.shape[0], 1)), X1], axis=1)
        S2 = np.dot(X1, W2)
        X2 = np.tanh(S2)

        delta2 = -2 * (Y_ - X2) * (1 - np.tanh(S2) ** 2)
        # 去掉对x0的求导
        delta1 = np.dot(delta2, W2.T)[:, 1:] * (1 - np.tanh(S1) ** 2)

        gradient_W1 = delta1 * X0.T
        gradient_W2 = delta2 * X1.T

        W1 = W1 - eta * gradient_W1
        W2 = W2 - eta * gradient_W2

    print()

    X0 = X_train

    X0 = np.concatenate([np.ones((X0.shape[0], 1)), X0], axis=1)
    S1 = np.dot(X0, W1)
    X1 = np.tanh(S1)

    X1 = np.concatenate([np.ones((X1.shape[0], 1)), X1], axis=1)
    S2 = np.dot(X1, W2)
    X2 = np.tanh(S2)
    print('Ein 0/1', np.average(Y_train != np.sign(X2)))

    X0 = X_test

    X0 = np.concatenate([np.ones((X0.shape[0], 1)), X0], axis=1)
    S1 = np.dot(X0, W1)
    X1 = np.tanh(S1)

    X1 = np.concatenate([np.ones((X1.shape[0], 1)), X1], axis=1)
    S2 = np.dot(X1, W2)
    X2 = np.tanh(S2)
    Eout = np.average(Y_test != np.sign(X2))
    print('Eout 0/1', Eout)

    return Eout


def backprop_stochastic_deeper(r, eta, T=50000):
    '''
    d - 8 - 3 - 1 neural network with tanh-type neurons, including the output neuron
    :return:
    '''

    train_raw = np.loadtxt('https://www.csie.ntu.edu.tw/~htlin/mooc/datasets/mltech/hw4_nnet_train.dat')
    test_raw = np.loadtxt('https://www.csie.ntu.edu.tw/~htlin/mooc/datasets/mltech/hw4_nnet_test.dat')

    X_train = train_raw[:, :-1]
    Y_train = train_raw[:, [-1]]
    X_test = test_raw[:, :-1]
    Y_test = test_raw[:, [-1]]

    X = X_train
    Y = Y_train

    N = X.shape[0]
    d = X.shape[1]

    d1 = 8
    d2 = 3

    # 注意要添加x0 = 1。如果不加，结果很差
    W1 = (2 * np.random.random((d + 1, d1)) - 1) * r
    W2 = (2 * np.random.random((d1 + 1, d2)) - 1) * r
    W3 = (2 * np.random.random((d2 + 1, 1)) - 1) * r

    for i in range(T):
        random_index = np.random.choice(N, 1)
        X0 = X[random_index, :]
        Y_ = Y[random_index]

        X0 = np.concatenate([np.ones((X0.shape[0], 1)), X0], axis=1)
        S1 = np.dot(X0, W1)
        X1 = np.tanh(S1)

        X1 = np.concatenate([np.ones((X1.shape[0], 1)), X1], axis=1)
        S2 = np.dot(X1, W2)
        X2 = np.tanh(S2)

        X2 = np.concatenate([np.ones((X2.shape[0], 1)), X2], axis=1)
        S3 = np.dot(X2, W3)
        X3 = np.tanh(S3)

        delta3 = -2 * (Y_ - X3) * (1 - np.tanh(S3) ** 2)
        # 去掉对x0的求导
        delta2 = np.dot(delta3, W3.T)[:, 1:] * (1 - np.tanh(S2) ** 2)
        delta1 = np.dot(delta2, W2.T)[:, 1:] * (1 - np.tanh(S1) ** 2)

        gradient_W1 = delta1 * X0.T
        gradient_W2 = delta2 * X1.T
        gradient_W3 = delta3 * X2.T

        W1 = W1 - eta * gradient_W1
        W2 = W2 - eta * gradient_W2
        W3 = W3 - eta * gradient_W3

    print()

    X0 = X_train
    X0 = np.concatenate([np.ones((X0.shape[0], 1)), X0], axis=1)
    S1 = np.dot(X0, W1)
    X1 = np.tanh(S1)

    X1 = np.concatenate([np.ones((X1.shape[0], 1)), X1], axis=1)
    S2 = np.dot(X1, W2)
    X2 = np.tanh(S2)

    X2 = np.concatenate([np.ones((X2.shape[0], 1)), X2], axis=1)
    S3 = np.dot(X2, W3)
    X3 = np.tanh(S3)

    print('Ein 0/1', np.average(Y_train != np.sign(X3)))

    X0 = X_test
    X0 = np.concatenate([np.ones((X0.shape[0], 1)), X0], axis=1)
    S1 = np.dot(X0, W1)
    X1 = np.tanh(S1)

    X1 = np.concatenate([np.ones((X1.shape[0], 1)), X1], axis=1)
    S2 = np.dot(X1, W2)
    X2 = np.tanh(S2)

    X2 = np.concatenate([np.ones((X2.shape[0], 1)), X2], axis=1)
    S3 = np.dot(X2, W3)
    X3 = np.tanh(S3)
    Eout = np.average(Y_test != np.sign(X3))
    print('Eout 0/1', Eout)

    return Eout


if __name__ == '__main__':
    Eout1 = np.average([backprop_stochastic(1, 0.1, 0.1) for i in range(10)])
    Eout2 = np.average([backprop_stochastic(6, 0.1, 0.1) for i in range(10)])
    Eout3 = np.average([backprop_stochastic(11, 0.1, 0.1) for i in range(10)])
    Eout4 = np.average([backprop_stochastic(16, 0.1, 0.1) for i in range(10)])
    Eout5 = np.average([backprop_stochastic(21, 0.1, 0.1) for i in range(10)])

    E6 = [backprop_stochastic(3, 0, 0.1) for i in range(10)]
    Eout6 = np.average(E6)
    E7 = [backprop_stochastic(3, 0.001, 0.1) for i in range(10)]
    Eout7 = np.average(E7)
    E8 = [backprop_stochastic(3, 0.1, 0.1) for i in range(10)]
    Eout8 = np.average(E8)
    E9 = [backprop_stochastic(3, 10, 0.1) for i in range(10)]
    Eout9 = np.average(E9)
    E10 = [backprop_stochastic(3, 1000, 0.1) for i in range(50)]
    Eout10 = np.average(E10)

    Eout11 = np.average([backprop_stochastic(3, 0.1, 0.001) for i in range(10)])
    Eout12 = np.average([backprop_stochastic(3, 0.1, 0.01) for i in range(10)])
    Eout13 = np.average([backprop_stochastic(3, 0.1, 0.1) for i in range(10)])
    Eout14 = np.average([backprop_stochastic(3, 0.1, 1) for i in range(10)])
    Eout15 = np.average([backprop_stochastic(3, 0.1, 10) for i in range(10)])

    Eout16 = np.average([backprop_stochastic_deeper(0.1, 0.01) for i in range(10)])

    print()
