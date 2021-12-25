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
    C=10000
)
clf.fit(X, Y)
print(clf.dual_coef_)

# solve with QP
from cvxopt import matrix, solvers

Q = np.dot(Y, Y.T) * (1 + np.dot(X, X.T)) ** 2
Q = matrix(Q.astype(float))

p = matrix(- np.ones((7, 1)))

a = -np.ones((7, 1))
a = a * np.identity(7)
a = matrix(a)

c = matrix(np.zeros((7, 1)))

# A = matrix(np.tile(Y, (1, 7)) * np.identity(7))
A = matrix(Y.T.astype(float))
b = matrix(np.zeros((1, 1)))

sol = solvers.qp(Q, p, a, c, A, b)

alpha = sol['x']

print(alpha)
print()
