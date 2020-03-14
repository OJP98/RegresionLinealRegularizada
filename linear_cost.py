import numpy as np


def linear_cost(X, y, theta, Lambda):
    m, _ = X.shape
    h = np.matmul(X, theta)
    sq = (y - h) ** 2
    theta = Lambda * (theta ** 2)
    return (theta.sum() + sq.sum()) / (2 * m)

    # return sq.sum() / (2 * m)
