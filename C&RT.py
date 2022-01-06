import numpy as np
from random import randrange


class DecisionTree:
    internal_nodes_count = 0

    def __init__(self, error):
        # 错误类型：回归、分类
        self.error = error
        # 终止条件满足时，返回常量，作为叶节点
        self.constant = None
        # 学习到的分支条件
        self.direction = None
        self.threshold = None
        self.feature_index = None
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
        x_unique = np.unique(X)
        if impurity == 0 or x_unique.shape[0] == 1:
            self.constant = optimal_constant(Y, self.error)
            return self

        self.internal_nodes_count += 1
        self.feature_index, self.threshold, self.direction, D1_mask, D2_mask = decision_stump_branch(X, Y,
                                                                                                     gini_impurity)
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
    N = X.shape[0]
    for feature_index in range(feature_num):
        x = X[:, [feature_index]]
        # 对当前单一特征维度去重
        x_unique = np.unique(x)
        # 对去重后的特征排序，要和原始的未排序的x分开
        x_sorted = np.sort(x_unique, axis=0)
        # 获取切割点阈值
        threshold_list = [(x_sorted[x_index, 0] + x_sorted[x_index + 1, 0]) / 2 for x_index in range(N - 1)]

        for threshold in threshold_list:
            for direction in [-1, 1]:
                D1_mask = X[:, [feature_index]] < threshold
                D2_mask = X[:, [feature_index]] > threshold

                D1_num = np.sum(D1_mask)
                D2_num = np.sum(D2_mask)

                Y1 = Y[D1_mask]
                Y2 = Y[D2_mask]

                impurity1 = impurity_function(Y1)
                impurity2 = impurity_function(Y2)

                weighted_impurity = D1_num * impurity1 + D2_num * impurity2

                if min_impurity is None or weighted_impurity < min_impurity:
                    min_impurity = weighted_impurity
                    feature_threshold_direction_list.append((feature_index, threshold, direction, D1_mask, D2_mask))

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
    if error == 'regression':
        return np.mean(Y)
    elif error == 'classification':
        return np.bincount(Y).argmax()

    raise Exception('undefined error type')


if __name__ == '__main__':
    train_raw = np.loadtxt('https://www.csie.ntu.edu.tw/~htlin/mooc/datasets/mltech/hw3_dectree_train.dat')
    test_raw = np.loadtxt('https://www.csie.ntu.edu.tw/~htlin/mooc/datasets/mltech/hw3_dectree_test.dat')

    X_train = train_raw[:, :-1]
    Y_train = train_raw[:, [-1]]
    X_test = test_raw[:, :-1]
    Y_test = test_raw[:, [-1]]

    t = DecisionTree('classification').fit(X_train, Y_train)
    t.batch_predict(X_train)
