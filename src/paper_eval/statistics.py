from __future__ import annotations
import numpy as np

def bootstrap_mean_ci(values, *, seed: int = 29, iterations: int = 1000, confidence: float = 0.95):
    array = np.asarray([float(v) for v in values if v is not None and np.isfinite(float(v))], dtype=float)
    if array.size == 0: return {"mean": None, "lower": None, "upper": None, "n": 0}
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(array, size=(iterations, array.size), replace=True), axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {"mean": float(array.mean()), "lower": float(np.quantile(means, alpha)), "upper": float(np.quantile(means, 1-alpha)), "n": int(array.size)}

def paired_bootstrap_difference(left, right, *, seed: int = 29, iterations: int = 1000, confidence: float = 0.95):
    pairs = [(float(a), float(b)) for a, b in zip(left, right) if a is not None and b is not None and np.isfinite(float(a)) and np.isfinite(float(b))]
    if not pairs: return {"mean_difference": None, "lower": None, "upper": None, "n": 0, "win_probability": None}
    differences = np.asarray([a-b for a,b in pairs], dtype=float)
    rng = np.random.default_rng(seed)
    sampled = np.mean(rng.choice(differences, size=(iterations, differences.size), replace=True), axis=1)
    alpha=(1-confidence)/2
    return {"mean_difference": float(differences.mean()), "lower": float(np.quantile(sampled, alpha)), "upper": float(np.quantile(sampled, 1-alpha)), "n": int(differences.size), "win_probability": float(np.mean(sampled > 0))}
