import time
import traceback
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

from .llm import LLMClient
from .data_loader import DataLoader
from .feature_executor import FeatureExecutor
from .evaluator import Evaluator
from .prompt_builder import PromptBuilder
from .memory import AgentMemory

load_dotenv()

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


class FeatureAgent:
    def __init__(self, time_budget: int = 575):
        self.time_budget = time_budget
        self.t0 = time.time()
        self.llm = LLMClient()
        self.loader = DataLoader(DATA_DIR)
        self.executor = FeatureExecutor()
        self.evaluator = Evaluator()
        self.memory = AgentMemory()

    def elapsed(self) -> float:
        return time.time() - self.t0

    def remaining(self) -> float:
        return self.time_budget - self.elapsed()

    def run(self):
        print(f"[Agent] Starting. Budget: {self.time_budget}s")
        datasets = self.loader.load()
        readme = self.loader.read_readme()
        train_df = datasets["train"]
        test_df = datasets["test"]
        target_col = self.loader.find_target(train_df)
        id_col = self.loader.find_id_col(train_df)
        y = train_df[target_col]

        print(f"[Agent] train={train_df.shape}, test={test_df.shape}")
        print(f"[Agent] target='{target_col}', id='{id_col}'")
        print(f"[Agent] Target distribution:\n{y.value_counts(normalize=True).to_string()}")

        data_context = self._build_context(datasets, readme, target_col, id_col)

        best_score = -1.0
        best_train_f = None
        best_test_f = None
        best_feat_names = None

        max_attempts = 5

        for attempt in range(1, max_attempts + 1):
            if self.remaining() < 80:
                print(f"[Agent] Time budget tight ({self.remaining():.0f}s left), stopping loop")
                break

            print(f"\n[Agent] == Attempt {attempt}/{max_attempts} "
                  f"(elapsed={self.elapsed():.0f}s, remaining={self.remaining():.0f}s) ==")

            try:

                temperature = 0.3 if attempt <= 2 else 0.7
                user_prompt = PromptBuilder.build(
                    data_context, attempt, self.memory.get_all()
                )
                code = self.llm.chat(PromptBuilder.SYSTEM, user_prompt, temperature=temperature)

                if not code or "def generate_features" not in code:
                    print("[Agent] LLM returned invalid code, skipping")
                    continue


                train_f, test_f, feat_names = self.executor.execute(
                    code, datasets, target_col, id_col
                )
                if train_f is None or not feat_names:
                    print("[Agent] Execution failed")
                    continue

                print(f"[Agent] Features generated: {feat_names}")

                # ── 4. Evaluate ───────────────────────────────────────────────
                score = self.evaluator.evaluate(train_f, y)
                print(f"[Agent] ROC-AUC = {score:.4f}  (best so far = {best_score:.4f})")
                self.memory.record(attempt, feat_names, score, best_score)

                if score > best_score:
                    best_score = score
                    best_train_f = train_f.copy()
                    best_test_f = test_f.copy()
                    best_feat_names = feat_names
                    print("[Agent] New best!")

            except Exception as e:
                print(f"[Agent] Attempt {attempt} crashed: {e}")
                traceback.print_exc()

        if best_train_f is not None and best_train_f.shape[1] > 5:
            print("\n[Agent] Running SHAP selection on best feature set...")
            top5 = self.evaluator.shap_select_top5(best_train_f, y)
            best_train_f = best_train_f[top5]
            best_test_f = best_test_f[top5]
            best_feat_names = top5
            final_score = self.evaluator.evaluate(best_train_f, y)
            print(f"[Agent] Score after SHAP selection: {final_score:.4f}")

        if best_train_f is not None:
            self._save(train_df, test_df, best_train_f, best_test_f,
                       target_col, id_col, best_feat_names)
            print(f"\n[Agent] Done! Best ROC-AUC={best_score:.4f}, "
                  f"features={best_feat_names}")
        else:
            print("[Agent] All attempts failed — saving numeric fallback")
            self._save_fallback(train_df, test_df, target_col, id_col)

        print(f"[Agent] Total time: {self.elapsed():.1f}s")


    def _build_context(self, datasets, readme, target_col, id_col) -> str:
        train = datasets["train"]
        parts = [
            f"README:\n{readme}",
            f"\nTARGET: '{target_col}'  |  ID: '{id_col}'",
            f"Train shape: {train.shape}  |  Test shape: {datasets['test'].shape}",
            f"\nColumn dtypes:\n{train.dtypes.to_string()}",
            f"\nDescriptive stats:\n{train.describe(include='all').to_string()}",
            f"\nMissing values:\n{train.isnull().sum()[train.isnull().sum() > 0].to_string() or 'None'}",
            f"\nFirst 5 rows:\n{train.head(5).to_string()}",
            self.loader.build_join_context(datasets, train, id_col),
        ]
        return "\n".join(parts)

    def _save(self, train_df, test_df, train_f, test_f, target_col, id_col, feat_names):
        feat_names = feat_names[:5]

        out_train = pd.DataFrame({
            id_col: train_df[id_col].values,
            target_col: train_df[target_col].values,
        })
        out_test = pd.DataFrame({id_col: test_df[id_col].values})

        for col in feat_names:
            out_train[col] = train_f[col].values
            out_test[col] = test_f[col].values

        out_train.to_csv(OUTPUT_DIR / "train.csv", index=False)
        out_test.to_csv(OUTPUT_DIR / "test.csv", index=False)
        print(f"[Agent] Saved: output/train.csv {out_train.shape}, "
              f"output/test.csv {out_test.shape}")

    def _save_fallback(self, train_df, test_df, target_col, id_col):
        num_cols = [
            c for c in train_df.columns
            if c not in (target_col, id_col)
            and pd.api.types.is_numeric_dtype(train_df[c])
        ][:5]

        out_train = train_df[[id_col, target_col] + num_cols].copy()
        out_test = test_df[[id_col] + num_cols].copy()
        out_train.to_csv(OUTPUT_DIR / "train.csv", index=False)
        out_test.to_csv(OUTPUT_DIR / "test.csv", index=False)
        print(f"[Agent] Fallback saved with cols: {num_cols}")
