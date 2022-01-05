import numpy as np
from random import randrange


class decision_tree():

    def __init__(self):
        self.


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
        # 要和原始的未排序的x分开
        x_sorted = np.sort(x, axis=0)
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
                    feature_threshold_direction_list.append((feature_index, threshold, direction))

    return feature_threshold_direction_list[randrange(len(feature_threshold_direction_list))]


def gini_impurity(Y):
    N = Y.shape[0]
    unique_y = np.unique(Y)
    purity = 0
    for y in unique_y:
        purity += (np.sum(Y == y) / N) ** 2
    impurity = 1 - purity
    return impurity
