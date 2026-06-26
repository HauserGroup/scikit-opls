import numpy as np


def make_regression_data(n_samples=80, n_features=30, n_ortho=2, amp=6.0, seed=0):
    """y-correlated signal plus large y-orthogonal structured variation."""
    rng = np.random.default_rng(seed)
    y = rng.normal(size=n_samples)
    y -= y.mean()
    p_pred = rng.normal(size=n_features)
    X = np.outer(y, p_pred)
    for _ in range(n_ortho):
        t_o = rng.normal(size=n_samples)
        t_o -= t_o.mean()
        t_o -= (t_o @ y) / (y @ y) * y  # exactly orthogonal to y
        p_o = amp * (0.5 * p_pred + rng.normal(size=n_features))
        X += np.outer(t_o, p_o)
    X += 0.05 * rng.normal(size=(n_samples, n_features))
    return X, y


def make_classification_data(n_per_class=40, n_features=30, n_ortho=2, amp=6.0, seed=0):
    """Two classes separated along one direction, plus class-orthogonal noise."""
    rng = np.random.default_rng(seed)
    n = 2 * n_per_class
    labels = np.array(["ctrl"] * n_per_class + ["case"] * n_per_class)
    sign = np.where(labels == "case", 1.0, -1.0)

    p_pred = rng.normal(size=n_features)
    X = np.outer(sign, p_pred)
    for _ in range(n_ortho):
        t_o = rng.normal(size=n)
        t_o -= t_o.mean()
        t_o -= (t_o @ sign) / (sign @ sign) * sign  # orthogonal to class
        p_o = amp * rng.normal(size=n_features)
        X += np.outer(t_o, p_o)
    X += 0.1 * rng.normal(size=(n, n_features))
    return X, labels
