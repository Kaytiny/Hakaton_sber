import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


class Evaluator:
    def evaluate(self, X: pd.DataFrame, y: pd.Series, n_splits: int = 3) -> float:
        """Cross-validated ROC-AUC with default CatBoost."""
        model = CatBoostClassifier(verbose=0, random_seed=42)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = []

        for train_idx, val_idx in skf.split(X, y):
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            proba = model.predict_proba(X.iloc[val_idx])[:, 1]
            scores.append(roc_auc_score(y.iloc[val_idx], proba))

        return float(np.mean(scores))

    def shap_select_top5(self, X: pd.DataFrame, y: pd.Series) -> list:
        """Train CatBoost, return top-5 features by importance."""
        if X.shape[1] <= 5:
            return list(X.columns)

        model = CatBoostClassifier(verbose=0, random_seed=42, iterations=300)
        model.fit(X, y)

        importances = model.get_feature_importance()
        ranked = sorted(zip(X.columns, importances), key=lambda x: -x[1])
        top5 = [col for col, _ in ranked[:5]]
        print(f"[Evaluator] SHAP top-5: {top5}")
        return top5
