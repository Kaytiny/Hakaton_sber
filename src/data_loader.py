import pandas as pd
import numpy as np
from pathlib import Path


class DataLoader:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir

    def read_readme(self) -> str:
        p = self.data_dir / "readme.txt"
        if p.exists():
            return p.read_text(encoding="utf-8", errors="replace")
        return "No readme found."

    def load(self) -> dict:
        datasets = {}
        for fpath in sorted(self.data_dir.glob("*.csv")):
            name = fpath.stem
            df = self._read_csv(fpath)
            datasets[name] = df
            print(f"[DataLoader] Loaded '{name}': {df.shape}")
        return datasets

    def _read_csv(self, path: Path) -> pd.DataFrame:
        for sep in (",", ";", "\t", "|"):
            try:
                df = pd.read_csv(path, sep=sep, low_memory=False)
                if df.shape[1] > 1:
                    return df
            except Exception:
                continue
        return pd.read_csv(path, low_memory=False)

    def find_target(self, train_df: pd.DataFrame) -> str:
        binary_candidates = []
        for col in train_df.columns:
            vals = set(train_df[col].dropna().unique())
            if vals.issubset({0, 1, 0.0, 1.0, "0", "1", True, False}):
                binary_candidates.append(col)

        for keyword in ("target", "label", "y", "class", "flag", "churn", "default"):
            for c in binary_candidates:
                if keyword in c.lower():
                    return c
        if binary_candidates:
            return binary_candidates[-1]
        for col in train_df.columns:
            if "target" in col.lower():
                return col
        raise ValueError("Cannot determine target column")

    def find_id_col(self, df: pd.DataFrame) -> str:
        for col in df.columns:
            if col.lower() in ("id", "index", "row_id", "sample_id",
                               "customer_id", "user_id", "object_id"):
                return col
        return df.columns[0]

    def build_join_context(self, datasets: dict, train_df: pd.DataFrame, id_col: str) -> str:
        """Analyse potential join keys between extra tables and train."""
        extra = {k: v for k, v in datasets.items() if k not in ("train", "test")}
        if not extra:
            return ""

        lines = ["\nAVAILABLE EXTRA TABLES FOR JOIN:"]
        train_cols = set(train_df.columns)

        for name, df in extra.items():
            lines.append(f"\nTable '{name}': shape={df.shape}")
            lines.append(f"  Columns: {list(df.columns)}")
            common = train_cols & set(df.columns)
            if common:
                lines.append(f"  Possible join keys with train: {list(common)}")
            lines.append(f"  Sample:\n{df.head(2).to_string()}")

        return "\n".join(lines)
