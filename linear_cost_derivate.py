import numpy as np


def linear_cost_derivate(X, y, theta, Lambda):
	h = np.matmul(X, theta)
	m, _ = X.shape
	mul = np.matmul((h - y).T, X).T
	c = Lambda * theta
	return (mul + c) / m
	# return np.matmul((h - y).T, X).T / m
