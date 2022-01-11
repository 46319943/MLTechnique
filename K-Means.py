import numpy as np


def K_Means(K):
    train_raw = np.loadtxt('https://www.csie.ntu.edu.tw/~htlin/mooc/datasets/mltech/hw4_nolabel_train.dat')

    X = train_raw

    N = X.shape[0]
    S = np.zeros((N, 1))

    init_index = np.random.choice(N, K)
    U = X[init_index]

    S_old = None

    while (np.any(S != S_old)):
        S_old = S.copy()

        # Optimize S
        distance_matrix = []
        for s in range(K):
            distance = np.sum(np.square(X - U[s]), axis=1)
            distance_matrix.append(distance)
        distance_matrix = np.vstack(distance_matrix).T
        S = np.argmin(distance_matrix, axis=1).reshape((-1, 1))

        # Optimize u
        for s in range(K):
            if X[(S == s).reshape(-1)].shape[0] != 0:
                U[s] = np.average(X[(S == s).reshape(-1)], axis=0)

    err_sum = 0
    for s in range(K):
        err_sum += np.sum(np.square(X[(S == s).reshape(-1)] - U[s]))
    Ein = err_sum / N

    print(Ein)
    return Ein

if __name__ == '__main__':
    Ein1 = np.average([K_Means(2) for i in range(10)])
    Ein2 = np.average([K_Means(10) for i in range(10)])
    print()