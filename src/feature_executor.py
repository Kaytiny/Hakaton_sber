import re
import traceback
import pandas as pd
import numpy as np


class FeatureExecutor:
    def execute(self, code: str, datasets: dict, target_col: str, id_col: str):
        code = self._clean_code(code)

        if "def generate_features" not in code:
            print("[Executor] No 'generate_features' function found")
            return None, None, []

        namespace = {"pd": pd, "np": np}

        try:
            exec(code, namespace)
        except Exception as e:
            print(f"[Executor] Compile error: {e}")
            traceback.print_exc()
            return None, None, []

        fn = namespace.get("generate_features")
        if fn is None:
            return None, None, []

        try:
            result = fn(datasets, target_col, id_col)
        except Exception as e:
            print(f"[Executor] Runtime error: {e}")
            traceback.print_exc()
            return None, None, []

        if not isinstance(result, tuple) or len(result) != 2:
            print("[Executor] Must return (train_df, test_df)")
            return None, None, []

        train_f, test_f = result
        if not isinstance(train_f, pd.DataFrame) or not isinstance(test_f, pd.DataFrame):
            return None, None, []
        if train_f.shape[1] == 0:
            return None, None, []

        for drop in [id_col, target_col]:
            if drop in train_f.columns:
                train_f = train_f.drop(columns=[drop])
            if drop in test_f.columns:
                test_f = test_f.drop(columns=[drop])

        common = [c for c in train_f.columns if c in test_f.columns][:5]
        if not common:
            return None, None, []

        train_f = self._sanitize(train_f[common].reset_index(drop=True))
        test_f = self._sanitize(test_f[common].reset_index(drop=True), ref=train_f)

        return train_f, test_f, common

    def _clean_code(self, code: str) -> str:
        code = re.sub(r"```[a-zA-Z]*\n?", "", code)
        return code.replace("```", "").strip()

    def _sanitize(self, df: pd.DataFrame, ref: pd.DataFrame = None) -> pd.DataFrame:
        df = df.copy().replace([np.inf, -np.inf], np.nan)
        for col in df.columns:
            if df[col].isnull().any():
                fill = (ref[col].median()
                        if ref is not None and col in ref.columns
                        else df[col].median())
                df[col] = df[col].fillna(fill if pd.notna(fill) else 0)
        return df
