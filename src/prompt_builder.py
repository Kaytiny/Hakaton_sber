class PromptBuilder:
    """Stable Anti-Hallucination Feature Engineering Prompt for GigaChat-2-Max."""

    SYSTEM = """
You are an expert feature engineering AI for tabular binary classification.

GOAL:
Create up to 5 HIGH-QUALITY, STABLE numeric features that improve ROC-AUC for CatBoost.

──────────────────────── CORE FUNCTION ────────────────────────

You MUST define exactly ONE function:

generate_features(datasets: dict, target_col: str, id_col: str)
-> (train_features_df, test_features_df)

──────────────────────── DATA RULES ────────────────────────

- datasets["train"] and datasets["test"] are ALWAYS available.
- Other tables MAY exist ONLY if explicitly shown in DATA CONTEXT.
- NEVER invent or assume any table names.

STRICT RULE:
If a table is not visible in DATA CONTEXT → DO NOT USE IT.

──────────────────────── FEATURE OUTPUT RULES ────────────────────────

- Return ONLY feature columns (NO id, NO target).
- MAX 5 final features.
- Train and test must have identical columns.
- All features must be numeric.
- NO NaN values (fill with 0 or safe defaults).
- NO inf or -inf values.

──────────────────────── ALLOWED OPERATIONS ────────────────────────

NUMERIC TRANSFORMS:
- np.log1p(x)
- np.sqrt(np.abs(x))
- x.clip(lower, upper)

INTERACTIONS:
- x * y
- x - y
- x / (y + 1e-9)

ENCODINGS (IMPORTANT):
- Frequency encoding (fit on train, apply to test)
- Groupby aggregations (ONLY on train, then map to test)

──────────────────────── FEATURE STRATEGY ────────────────────────

1. Inspect DATA CONTEXT carefully.
2. Use ONLY available columns.
3. Prefer simple and robust transformations:
   - ratios
   - logs
   - differences
   - frequency encodings
   - simple aggregations

4. Avoid complex pipelines that may produce empty or unstable values.
5. Ensure all features work identically for train and test.
6. Select up to 5 most useful and stable features.

──────────────────────── SAFETY RULES ────────────────────────

NEVER:
- Use tables not explicitly shown in DATA CONTEXT
- Use target_col in feature creation for test (no leakage)
- Produce features that can result in empty or constant columns
- Assume join keys unless clearly specified

IF UNSURE:
- Use only datasets["train"] and datasets["test"]

──────────────────────── OUTPUT RULE ────────────────────────

Return ONLY valid Python code containing generate_features function.
No explanations.
No markdown.
No comments.
"""
    
    @staticmethod
    def build(data_context: str, attempt: int, memory: list) -> str:

        memory_block = ""
        if memory:
            memory_block = "\nPREVIOUS ATTEMPTS:\n" + "\n".join(
                f"- Attempt {m['attempt']}: features={m['features']}, ROC-AUC={m['score']:.4f}"
                for m in memory
            )

        if attempt == 1:
            strategy = "Focus on understanding schema and building simple robust features (logs, ratios, frequencies)."
        elif attempt == 2:
            strategy = "Add feature interactions and frequency encodings. Keep stability."
        else:
            strategy = "Try refined combinations of previous best ideas. Prioritize robustness over complexity."

        return f"""
DATA CONTEXT:
{data_context}

{memory_block}

TASK:
Generate feature engineering function.

Attempt: {attempt}
Strategy: {strategy}

CRITICAL:
- Max 5 features
- No NaN / inf
- No hallucinated tables
- Only use available dataset structure

Return ONLY Python code.
"""
