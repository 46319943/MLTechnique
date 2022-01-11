import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm


def linear_soft_svm():
    train_raw = np.loadtxt('http://www.amlbook.com/data/zip/features.train')
    test_raw = np.loadtxt('http://www.amlbook.com/data/zip/features.test')

    train_raw[train_raw[:, 0] != 0, 0] = 1
    test_raw[test_raw[:, 0] != 0, 0] = 1

    train_raw_X = train_raw[:, [1, 2]]
    train_raw_Y = train_raw[:, [0]]

    X_train, X_val, Y_train, Y_val = train_test_split(train_raw_X, train_raw_Y)

    clf = svm.SVC(
        kernel='linear',
        C=0.01
    )
    clf.fit(X_train, Y_train)

    # clf.dual_coef_ (1, 70)
    # clf.support_vectors_ (70, 2)
    w = np.dot(clf.dual_coef_, clf.support_vectors_).T

    w_length = np.sqrt(np.dot(w.T, w))
    return w_length


def polynomial_kernel_soft_svm(true_label=0):
    train_raw = np.loadtxt('http://www.amlbook.com/data/zip/features.train')
    test_raw = np.loadtxt('http://www.amlbook.com/data/zip/features.test')

    train_true_label_index = train_raw[:, 0] == true_label
    test_true_label_index = test_raw[:, 0] == true_label
    train_raw[train_true_label_index, 0] = 1
    train_raw[~train_true_label_index, 0] = 0
    test_raw[test_true_label_index, 0] = 1
    test_raw[~test_true_label_index, 0] = 0

    train_raw_X = train_raw[:, [1, 2]]
    train_raw_Y = train_raw[:, [0]]

    # X_train, X_val, Y_train, Y_val = train_test_split(train_raw_X, train_raw_Y)

    # To get correct answer of alpha which depends on the support vectors (therefore training set)
    X_train = train_raw_X
    Y_train = train_raw_Y

    clf = svm.SVC(
        kernel='poly',
        C=0.01,
        gamma=1, coef0=1, degree=2
    )
    clf.fit(X_train, Y_train)
    Y_hat = clf.predict(X_train).reshape(-1, 1)
    E_01 = np.sum(Y_hat != Y_train) / X_train.shape[0]
    print(E_01)

    alpha_sum = np.sum(np.abs(clf.dual_coef_))
    print(alpha_sum)

    print()


def gaussian_soft_svm(c=0.01, gamma=100):
    train_raw = np.loadtxt('http://www.amlbook.com/data/zip/features.train')
    test_raw = np.loadtxt('http://www.amlbook.com/data/zip/features.test')

    train_raw[train_raw[:, 0] != 0, 0] = 1
    test_raw[test_raw[:, 0] != 0, 0] = 1

    train_raw_X = train_raw[:, [1, 2]]
    train_raw_Y = train_raw[:, [0]]

    X_train, X_val, Y_train, Y_val = train_test_split(train_raw_X, train_raw_Y)
    X_test = test_raw[:, [1, 2]]
    Y_test = test_raw[:, [0]]

    clf = svm.SVC(
        kernel='rbf',
        gamma=gamma,
        C=c
    )
    clf.fit(X_train, Y_train)

    Y_hat = clf.predict(X_train).reshape(-1, 1)
    E_01 = np.sum(Y_hat != Y_train) / X_train.shape[0]
    print(E_01)

    Y_hat_out = clf.predict(X_test).reshape(-1, 1)
    E_out = np.sum(Y_hat_out != Y_test) / X_test.shape[0]
    print(E_out)

    print()


def gaussian_soft_svm_validate():
    train_raw = np.loadtxt('http://www.amlbook.com/data/zip/features.train')
    test_raw = np.loadtxt('http://www.amlbook.com/data/zip/features.test')

    train_raw[train_raw[:, 0] != 0, 0] = 1
    test_raw[test_raw[:, 0] != 0, 0] = 1

    train_raw_X = train_raw[:, [1, 2]]
    train_raw_Y = train_raw[:, [0]]

    X_train, X_val, Y_train, Y_val = train_test_split(train_raw_X, train_raw_Y, )
    X_test = test_raw[:, [1, 2]]
    Y_test = test_raw[:, [0]]

    min_E_val = 1
    min_gamma = None
    c = 0.1
    for gamma in [1, 10, 100, 1000, 10000]:
        clf = svm.SVC(
            kernel='rbf',
            gamma=gamma,
            C=c
        )
        clf.fit(X_train, Y_train)

        Y_hat = clf.predict(X_train).reshape(-1, 1)
        E_01 = np.sum(Y_hat != Y_train) / X_train.shape[0]
        print(E_01)

        Y_hat_val = clf.predict(X_val).reshape(-1, 1)
        E_val = np.sum(Y_hat_val != Y_val) / X_val.shape[0]
        print(E_val)

        if E_val < min_E_val:
            min_gamma = gamma
            min_E_val = E_val

    print()
    return min_gamma


if __name__ == '__main__':
    # linear_soft_svm()
    polynomial_kernel_soft_svm(0)
    polynomial_kernel_soft_svm(2)
    polynomial_kernel_soft_svm(4)
    polynomial_kernel_soft_svm(6)
    polynomial_kernel_soft_svm(8)
    # gaussian_soft_svm(c=0.1, gamma=1)
    # gaussian_soft_svm(c=0.1, gamma=10)
    # gaussian_soft_svm(c=0.1, gamma=100)
    # gaussian_soft_svm(c=0.1, gamma=1000)
    # gaussian_soft_svm(c=0.1, gamma=10000)

    # gamma_list = []
    # for i in range(100):
    #     gamma_list.append(
    #         gaussian_soft_svm_validate()
    #     )
    print()
