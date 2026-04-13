"""Check submission requirements before sending zip."""
import sys
from pathlib import Path


def check():
    errors = []
    warnings = []

    # 1. pyproject.toml
    if not Path("pyproject.toml").exists():
        errors.append("Missing pyproject.toml")

    # 2. run.py
    if not Path("run.py").exists():
        errors.append("Missing run.py")

    # 3. .env
    if not Path(".env").exists():
        errors.append("Missing .env")
    else:
        env_text = Path(".env").read_text()
        if "GIGACHAT_CREDENTIALS" not in env_text:
            errors.append(".env missing GIGACHAT_CREDENTIALS")
        if "GIGACHAT_SCOPE" not in env_text:
            errors.append(".env missing GIGACHAT_SCOPE")
        if "ВАШ_ТОКЕН_ЗДЕСЬ" in env_text:
            warnings.append(".env still has placeholder token — replace before submit")

    if not Path("data").exists():
        errors.append("Missing data/ directory")
    else:
        for fname in ("readme.txt", "train.csv", "test.csv"):
            if not (Path("data") / fname).exists():
                errors.append(f"Missing data/{fname}")

    # 5. src/
    for fpath in (
        "src/__init__.py",
        "src/agent.py",
        "src/llm.py",
        "src/data_loader.py",
        "src/feature_executor.py",
        "src/evaluator.py",
        "src/prompt_builder.py",
        "src/memory.py",
    ):
        if not Path(fpath).exists():
            errors.append(f"Missing {fpath}")

    if Path("output").exists():
        for fname in ("train.csv", "test.csv"):
            fpath = Path("output") / fname
            if not fpath.exists():
                errors.append(f"Missing output/{fname} — run 'python run.py' first")
            else:
                import pandas as pd
                df = pd.read_csv(fpath)
                if df.shape[1] < 2:
                    errors.append(f"output/{fname} has fewer than 2 columns")
                if df.shape[0] == 0:
                    errors.append(f"output/{fname} is empty")


        if (Path("output") / "train.csv").exists() and (Path("output") / "test.csv").exists():
            import pandas as pd
            tr = pd.read_csv(Path("output") / "train.csv")
            te = pd.read_csv(Path("output") / "test.csv")
            tr_feats = set(tr.columns[2:])  # skip id + target
            te_feats = set(te.columns[1:])  # skip id
            if tr_feats != te_feats:
                errors.append(
                    f"Feature columns mismatch between train and test output:\n"
                    f"  train extra: {tr_feats - te_feats}\n"
                    f"  test extra:  {te_feats - tr_feats}"
                )
            n_feats = len(tr_feats)
            if n_feats > 5:
                errors.append(f"output/train.csv has {n_feats} features — max is 5")
    else:
        warnings.append("output/ not found — run 'python run.py' first, then re-check")

    print()
    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(f"  ⚠  {w}")
    if errors:
        print("\nFAILED:")
        for e in errors:
            print(f"  ✗  {e}")
        sys.exit(1)
    else:
        print("All checks passed! Ready to submit.")


if __name__ == "__main__":
    check()
