import numpy as np
from random import randrange

g_Ein_list = []


class RandomForest:

    def __init__(self, T, N_apostrophe):
        self.T = T
        self.N_apostrophe = N_apostrophe
        self.g_list = []

    def fit(self, X, Y):
        N = X.shape[0]

        for t in range(self.T):
            sample_mask = np.random.choice(N, self.N_apostrophe)
            X_sample = X[sample_mask]
            Y_sample = Y[sample_mask]
            g = DecisionTree('classification').fit(X_sample, Y_sample)
            self.g_list.append(g)

            g_Ein_list.append(
                np.average(Y != g.batch_predict(X))
            )

        return self

    def __call__(self, X):
        N = X.shape[0]
        Y = np.zeros((N, 1))
        for g in self.g_list:
            Y += g.batch_predict(X)
        return Y / N


class DecisionTree:
    internal_nodes_count = 0

    def __init__(self, error):
        # 错误类型：回归、分类
        self.error = error
        # 终止条件满足时，返回常量，作为叶节点
        self.constant = None
        # 学习到的分支条件
        self.feature_index = None
        self.threshold = None
        # 分支后的数据
        self.X1 = None
        self.X2 = None
        self.Y1 = None
        self.Y2 = None
        # 子树
        self.subtree1 = None
        self.subtree2 = None

        return

    def fit(self, X, Y):
        # check termination criteria
        impurity = gini_impurity(Y)
        x_unique = np.unique(X, axis=0)
        if impurity == 0 or x_unique.shape[0] == 1:
            self.constant = optimal_constant(Y, self.error)
            return self

        DecisionTree.internal_nodes_count += 1
        self.feature_index, self.threshold, D1_mask, D2_mask = decision_stump_branch(X, Y, gini_impurity)
        self.X1 = X[D1_mask]
        self.X2 = X[D2_mask]
        self.Y1 = Y[D1_mask]
        self.Y2 = Y[D2_mask]

        self.subtree1 = DecisionTree(self.error).fit(self.X1, self.Y1)
        self.subtree2 = DecisionTree(self.error).fit(self.X2, self.Y2)

        return self

    def __call__(self, x):
        if self.constant is not None:
            return self.constant

        if x[:, self.feature_index] < self.threshold:
            return self.subtree1(x)
        else:
            return self.subtree2(x)

    def batch_predict(self, X):
        Y = np.ones((X.shape[0], 1))

        if self.constant is not None:
            return Y * self.constant

        branch_mask = X[:, self.feature_index] < self.threshold
        Y[branch_mask] = self.subtree1.batch_predict(X[branch_mask])
        Y[~branch_mask] = self.subtree2.batch_predict(X[~branch_mask])
        return Y


def decision_stump_branch(X, Y, impurity_function):
    '''
    对于h(x)=0或者h(x)=1，说明yn相同，impurity为0，不需要再分了，不会进入到这一步，所以不考虑这个情况
    :param X:
    :param Y:
    :param example_weight:
    :return:
    '''

    min_impurity = None
    feature_threshold_direction_list = []

    feature_num = X.shape[1]

    for feature_index in range(feature_num):
        x = X[:, [feature_index]]
        # 对当前单一特征维度去重
        x_unique = np.unique(x, axis=0)
        # 对去重后的特征排序，要和原始的未排序的x分开
        x_sorted = np.sort(x_unique, axis=0)
        # 去重后重新计算特征值数量
        N = x_sorted.shape[0]
        # 获取切割点阈值
        threshold_list = [(x_sorted[x_index, 0] + x_sorted[x_index + 1, 0]) / 2 for x_index in range(N - 1)]

        for threshold in threshold_list:
            # 在进行branching的时候，不需要考虑方向，因为会得到相同的impurity

            D1_mask = X[:, [feature_index]] < threshold
            D2_mask = X[:, [feature_index]] > threshold
            D1_mask = D1_mask.reshape(-1)
            D2_mask = D2_mask.reshape(-1)

            D1_num = np.sum(D1_mask)
            D2_num = np.sum(D2_mask)

            Y1 = Y[D1_mask]
            Y2 = Y[D2_mask]

            impurity1 = impurity_function(Y1)
            impurity2 = impurity_function(Y2)

            weighted_impurity = D1_num * impurity1 + D2_num * impurity2

            if min_impurity is None or weighted_impurity < min_impurity:
                min_impurity = weighted_impurity
                feature_threshold_direction_list.clear()
                feature_threshold_direction_list.append((feature_index, threshold, D1_mask, D2_mask))

    return feature_threshold_direction_list[randrange(len(feature_threshold_direction_list))]


def gini_impurity(Y):
    N = Y.shape[0]
    unique_y = np.unique(Y)
    purity = 0
    for y in unique_y:
        purity += (np.sum(Y == y) / N) ** 2
    impurity = 1 - purity
    return impurity


def optimal_constant(Y, error):
    '''
    use 0/1 error for binary/multiclass classification: majority of Y
    use squared error for regression: average of Y
    :param Y:
    :param error:
    :return:
    '''
    Y = Y.reshape(-1)
    if error == 'regression':
        return np.mean(Y)
    elif error == 'classification':
        # This method only applies to positive integer
        # return np.bincount(Y).argmax()

        values, counts = np.unique(Y, return_counts=True)
        return values[np.argmax(counts)]

    raise Exception('undefined error type')


if __name__ == '__main__':
    train_raw = np.loadtxt('https://www.csie.ntu.edu.tw/~htlin/mooc/datasets/mltech/hw3_dectree_train.dat')
    test_raw = np.loadtxt('https://www.csie.ntu.edu.tw/~htlin/mooc/datasets/mltech/hw3_dectree_test.dat')

    X_train = train_raw[:, :-1]
    Y_train = train_raw[:, [-1]]
    X_test = test_raw[:, :-1]
    Y_test = test_raw[:, [-1]]

    N = X_train.shape[0]

    # t = DecisionTree('classification').fit(X_train, Y_train)
    #
    # Y_hat_train = t.batch_predict(X_train)
    # Y_hat_test = t.batch_predict(X_test)
    # Ein = np.average(Y_hat_train != Y_train)
    # Eout = np.average(Y_hat_test != Y_test)

    f = RandomForest(300, N).fit(X_train, Y_train)
    Y_hat_train = f(X_train)
    Y_hat_test = f(X_test)

    np.average(np.sign(Y_hat_train) != Y_train)
    np.average(np.sign(Y_hat_test) != Y_test)

    print()
