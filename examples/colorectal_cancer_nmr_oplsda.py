"""Binary OPLS-DA example using the colorectal cancer NMR dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
)
from sklearn.model_selection import train_test_split

from scikit_opls import OPLSDA
from scikit_opls.inspection import vip

DATA_URL = (
    "https://github.com/HauserGroup/scikit-opls/releases/download/data/"
    "colorectal_cancer_nmr.csv"
)
CLASSES = ["Healthy Control", "Colorectal Cancer"]
BALANCE_RANDOM_STATE = 2


def main() -> None:
    """Fit OPLS-DA and report held-out classification metrics."""
    data = pd.read_csv(DATA_URL)
    data = data[data["classification"].isin(CLASSES)]
    samples_per_class = data["classification"].value_counts().min()
    data = data.groupby("classification", group_keys=False).sample(
        n=samples_per_class,
        random_state=BALANCE_RANDOM_STATE,
    )

    feature_columns = [
        column
        for column in data.columns
        if column not in {"sample_id", "classification"}
    ]
    X = data[feature_columns].to_numpy(dtype=float)
    y = data["classification"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=0,
        stratify=y,
    )

    model = OPLSDA(n_components=1, n_orthogonal=2, scale="standard").fit(
        X_train,
        y_train,
    )
    predictions = model.predict(X_test)

    vip_order = np.argsort(vip(model))[-5:][::-1]
    important_bins = [feature_columns[index].strip() for index in vip_order]

    print(f"Balanced samples: {len(data)} ({samples_per_class} per class)")
    print(f"Spectral bins: {X.shape[1]}")
    print(f"Orthogonal components: {model.n_orthogonal_}")
    print(f"Test accuracy: {accuracy_score(y_test, predictions):.3f}")
    print(f"Test balanced accuracy: {balanced_accuracy_score(y_test, predictions):.3f}")
    print("Top predictive VIP bins:", ", ".join(important_bins))
    print()
    print(classification_report(y_test, predictions))


if __name__ == "__main__":
    main()
