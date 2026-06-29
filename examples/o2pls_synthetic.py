"""Synthetic two-block O2PLS example."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import r2_score

from scikit_opls import O2PLS


def _make_blocks(seed: int = 0):
    """Generate two blocks with joint and block-specific latent structure."""
    rng = np.random.default_rng(seed)
    n_samples = 100
    n_joint = 2
    latent, _ = np.linalg.qr(rng.normal(size=(n_samples, 4)))
    joint = latent[:, :n_joint] * np.array([5.0, 2.0])
    x_specific = latent[:, 2:3] * 4.0
    y_specific = latent[:, 3:4] * 3.0

    x_basis, _ = np.linalg.qr(rng.normal(size=(12, 3)))
    y_basis, _ = np.linalg.qr(rng.normal(size=(8, 3)))
    x_joint = joint @ x_basis[:, :n_joint].T
    y_joint = joint @ y_basis[:, :n_joint].T
    X = x_joint + x_specific @ x_basis[:, 2:3].T
    Y = y_joint + y_specific @ y_basis[:, 2:3].T
    X += 0.01 * rng.normal(size=X.shape)
    Y += 0.01 * rng.normal(size=Y.shape)
    return X, Y, x_joint, y_joint


def main() -> None:
    """Fit O2PLS and report joint-structure recovery on synthetic data."""
    X, Y, x_joint, y_joint = _make_blocks()
    model = O2PLS(
        n_components=2,
        n_x_orthogonal=1,
        n_y_orthogonal=1,
        scale="center",
    ).fit(X, Y)

    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")
    print(f"X-orthogonal components: {model.n_x_orthogonal_}")
    print(f"Y-orthogonal components: {model.n_y_orthogonal_}")
    print(f"R2X joint / orthogonal: {model.r2x_:.3f} / {model.r2x_ortho_:.3f}")
    print(f"R2Y joint / orthogonal: {model.r2y_:.3f} / {model.r2y_ortho_:.3f}")
    print(f"Predicted joint Y R2: {r2_score(y_joint, model.predict(X)):.3f}")
    print(f"Predicted joint X R2: {r2_score(x_joint, model.predict_x(Y)):.3f}")


if __name__ == "__main__":
    main()
