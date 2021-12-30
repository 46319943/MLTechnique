'''
solve kernel dual svm with sklearn and QP solver
'''

from sklearn import svm
import numpy as np

X = np.array([
    [1, 0],
    [0, 1],
    [0, -1],
    [-1, 0],
    [0, 2],
    [0, -2],
    [-2, 0]
])

Y = np.array([
    -1,
    -1,
    -1,
    1,
    1,
    1,
    1
]).reshape((-1, 1))

# solve with sklearn

clf = svm.SVC(
    kernel='poly',
    degree=2, coef0=1, gamma=1,
    # without Regularization. It should be linear separable
    C=1
)
clf.fit(X, Y)

## y_n * alpha_n
print(clf.dual_coef_)
## alpha_n >= 0, alpha_n = abs(dual_coef_)
alpha_ = abs(clf.dual_coef_)
print(alpha_, np.sum(alpha_))
## SV
print(clf.support_vectors_)
## b
print(clf.intercept_)

# solve with QP
from cvxopt import matrix, solvers

N = X.shape[0]
# Kernel Matrix
P = np.dot(Y, Y.T) * (1 + np.dot(X, X.T)) ** 2
P = matrix(P.astype(float))

q = matrix(- np.ones((N, 1)))

# alpha_n >= 0 ==> - alpha <= 0
G = - np.ones((N, 1))
G = G * np.identity(N)
G = matrix(G)
h = matrix(np.zeros((N, 1)))

# sum(y_n * alpha_n) = 0
A = matrix(Y.T.astype(float))
b = matrix(np.zeros((1, 1)))

sol = solvers.qp(P, q, G, h, A, b)

alpha = sol['x']

print(alpha, sum(alpha))
print()

# weight
# alpha * Y *
