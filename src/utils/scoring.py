"""Scoring script — reference implementation."""
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from pathlib import Path


def score_submission(output_dir: str = "output", data_dir: str = "data"):
    output_dir = Path(output_dir)

    train = pd.read_csv(output_dir / "train.csv")

    target_col = None
    for col in train.columns:
        vals = set(train[col].dropna().unique())
        if vals.issubset({0, 1, 0.0, 1.0}):
            target_col = col
            break

    if target_col is None:
        raise ValueError("Cannot find target column in output/train.csv")

    id_col = train.columns[0]
    feature_cols = [c for c in train.columns if c not in (id_col, target_col)]

    X = train[feature_cols]
    y = train[target_col]

    model = CatBoostClassifier(verbose=0, random_seed=42)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model.fit(X_tr, y_tr)

    proba = model.predict_proba(X_val)[:, 1]
    score = roc_auc_score(y_val, proba)
    print(f"Validation ROC-AUC: {score:.4f}")
    return score


if __name__ == "__main__":
    score_submission()
