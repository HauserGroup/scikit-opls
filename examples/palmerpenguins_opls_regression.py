"""OPLS regression example using the Palmer penguins dataset."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

from scikit_opls import OPLS

DATA_URL = (
    "https://github.com/HauserGroup/scikit-opls/releases/download/data/"
    "palmerpenguins.csv"
)
FEATURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm"]
TARGET = "body_mass_g"


def main() -> None:
    """Fit OPLS and report held-out body-mass regression metrics."""
    data = pd.read_csv(DATA_URL).dropna(subset=FEATURES + [TARGET])

    X = data[FEATURES].to_numpy(dtype=float)
    y = data[TARGET].to_numpy(dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=0,
    )

    model = OPLS(n_components=1, n_orthogonal=2, scale="standard").fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"Samples: {len(data)}")
    print("Features:", ", ".join(FEATURES))
    print(f"Orthogonal components: {model.n_orthogonal_}")
    print(f"Training R2Y: {model.r2y_:.3f}")
    print(f"Test R2: {r2_score(y_test, predictions):.3f}")
    print(f"Test RMSE: {root_mean_squared_error(y_test, predictions):.1f} g")
    print()
    print("First five held-out predictions:")
    for actual, predicted in zip(y_test[:5], predictions[:5], strict=False):
        print(f"  actual={actual:6.1f} g  predicted={predicted:6.1f} g")


if __name__ == "__main__":
    main()
