import numpy as np


def calculate_gncc(fixed, moving):

    mu_f = np.mean(fixed)
    mu_m = np.mean(moving)
    a = np.sum(abs(fixed - mu_f) * abs(moving - mu_m))
    b = np.sqrt(np.sum((fixed - mu_f) ** 2) * np.sum((moving - mu_m) ** 2))
    return a / b


def calculate_ncc(fixed, moving):
    assert fixed.shape == moving.shape

    med_f = np.median(np.max(fixed, axis=2))
    med_m = np.median(np.max(moving, axis=2))
    fixed_new = np.maximum(fixed - med_f, 0)
    moving_new = np.maximum(moving - med_m, 0)

    mu_f = np.mean(fixed_new)
    mu_m = np.mean(moving_new)
    fixed_new = fixed_new / mu_f - 1
    moving_new = moving_new / mu_m - 1
    numerator = np.sum(fixed_new * moving_new)
    denominator = np.sqrt(np.sum(fixed_new ** 2) * np.sum(moving_new ** 2))

    return numerator / denominator
