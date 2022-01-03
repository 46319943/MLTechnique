import numpy as np
from random import randrange


def adaboost_stump():
    train_raw = np.loadtxt('https://www.csie.ntu.edu.tw/~htlin/mooc/datasets/mltech/hw2_adaboost_train.dat')
    test_raw = np.loadtxt('https://www.csie.ntu.edu.tw/~htlin/mooc/datasets/mltech/hw2_adaboost_test.dat')

    X_train = train_raw[:, :-1]
    Y_train = train_raw[:, [-1]]
    X_test = test_raw[:, :-1]
    Y_test = test_raw[:, [-1]]

    N = X_train.shape[0]

    init_weight = np.ones((N, 1))
    scale_weight = init_weight
    aggregation_list = []
    for i in range(300):
        decision_stump_result = decision_stump(X_train, Y_train, scale_weight)
        feature_index, threshold, direction, scale_weight, alpha = decision_stump_result
        aggregation_list.append(decision_stump_result)

    y_hat = aggregation_decision_stump(aggregation_list, X_train)
    E_in = np.average(y_hat != Y_train)

    y_hat_test = aggregation_decision_stump(aggregation_list, X_test)
    E_out = np.average(y_hat_test != Y_test)

    print()


def decision_stump(X, Y, example_weight):
    min_Ein = 1
    feature_threshold_direction_list = []
    example_weight = np.array(example_weight).copy()

    feature_num = X.shape[1]
    N = X.shape[0]
    for feature_index in range(feature_num):
        x = X[:, [feature_index]]
        # 要和原始的未排序的x分开
        x_sorted = np.sort(x, axis=0)
        threshold_list = [None] + [(x_sorted[x_index, 0] + x_sorted[x_index + 1, 0]) / 2 for x_index in range(N - 1)]
        for threshold in threshold_list:
            for direction in [-1, 1]:
                if threshold is None:
                    y_hat = direction * np.ones((N, 1))
                else:
                    y_hat = direction * np.sign(x - threshold)

                correct_index = y_hat == Y
                Ein = np.average(example_weight * (y_hat != Y))
                epsilon = Ein / np.average(example_weight)
                scaling_factor = np.sqrt((1 - epsilon) / epsilon)
                scale_weight = np.copy(example_weight)

                # 对错的乘除写反了。在Ein小于0.5的时候，应该减少对的权重，增加错的权重
                scale_weight[correct_index] = scale_weight[correct_index] / scaling_factor
                scale_weight[~correct_index] = scale_weight[~correct_index] * scaling_factor
                alpha = np.log(scaling_factor)

                if Ein < min_Ein:
                    min_Ein = Ein
                    feature_threshold_direction_list.clear()
                    feature_threshold_direction_list.append((feature_index, threshold, direction, scale_weight, alpha))
                elif Ein == min_Ein:
                    feature_threshold_direction_list.append((feature_index, threshold, direction, scale_weight, alpha))

    return feature_threshold_direction_list[randrange(len(feature_threshold_direction_list))]


def aggregation_decision_stump(aggregation_list, X):
    N = X.shape[0]
    weighted_y_hat = np.zeros((N, 1))
    for decision_stump_result in aggregation_list:
        feature_index, threshold, direction, scale_weight, alpha = decision_stump_result
        x = X[:, [feature_index]]
        if threshold is None:
            y_hat = direction * np.ones((N, 1))
        else:
            y_hat = direction * np.sign(x - threshold)
        weighted_y_hat += alpha * y_hat
    return np.sign(weighted_y_hat)


if __name__ == '__main__':
    adaboost_stump()
