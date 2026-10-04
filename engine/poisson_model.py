"""Dixon-Coles adjusted Poisson scoreline model."""

import numpy as np
from scipy.stats import poisson as poisson_dist


def dixon_coles_tau(x, y, lam_h, lam_a, rho):
    if x == 0 and y == 0:
        return 1 - lam_h * lam_a * rho
    if x == 1 and y == 0:
        return 1 + lam_a * rho
    if x == 0 and y == 1:
        return 1 + lam_h * rho
    if x == 1 and y == 1:
        return 1 - rho
    return 1.0


def score_probability_matrix(lam_h, lam_a, rho, max_goals):
    matrix = np.zeros((max_goals, max_goals))
    for i in range(max_goals):
        for j in range(max_goals):
            p_ind = poisson_dist.pmf(i, lam_h) * poisson_dist.pmf(j, lam_a)
            tau = dixon_coles_tau(i, j, lam_h, lam_a, rho)
            matrix[i, j] = p_ind * tau
    matrix /= matrix.sum()
    return matrix


def extract_probabilities(matrix):
    n = matrix.shape[0]
    p_home = sum(matrix[i, j] for i in range(n) for j in range(n) if i > j)
    p_draw = sum(matrix[i, i] for i in range(n))
    p_away = sum(matrix[i, j] for i in range(n) for j in range(n) if i < j)
    return p_home, p_draw, p_away


def top_scorelines(matrix, n=10):
    mg = matrix.shape[0]
    scores = [(i, j, matrix[i, j]) for i in range(mg) for j in range(mg)]
    scores.sort(key=lambda x: x[2], reverse=True)
    return scores[:n]
