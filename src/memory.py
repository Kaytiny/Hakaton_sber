class AgentMemory:
    """Stores results of previous attempts and generates LLM feedback insights."""

    def __init__(self):
        self._history: list = []

    def record(self, attempt: int, features: list, score: float, prev_best: float):
        delta = score - prev_best if prev_best >= 0 else 0.0
        if delta > 0.005:
            insight = f"Improvement +{delta:.4f}. These feature types worked well."
        elif delta > 0:
            insight = f"Marginal improvement +{delta:.4f}. Try more aggressive transformations."
        elif delta > -0.005:
            insight = "No improvement. Avoid similar features, try a completely different approach."
        else:
            insight = f"Regression {delta:.4f}. These features hurt. Avoid this strategy entirely."

        self._history.append({
            "attempt": attempt,
            "features": features,
            "score": score,
            "insight": insight,
        })

    def get_all(self) -> list:
        return list(self._history)

    def best_score(self) -> float:
        if not self._history:
            return -1.0
        return max(m["score"] for m in self._history)
