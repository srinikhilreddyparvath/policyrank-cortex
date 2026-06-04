"""
MVP 24.3: Spell Check + Query Normalization Agent

Deterministic commerce-query cleanup that preserves hard constraints,
compatibility language, brand/model tokens, and numeric/model terms.
"""

from __future__ import annotations

import argparse
import csv
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from src.query_understanding_agent import console_text, load_all_queries, select_queries


OUTPUT_DIR = Path("outputs/query_normalization")

NEGATION_TERMS = {"without", "no", "not", "non", "excluding", "except"}
COMPATIBILITY_TERMS = {"for", "compatible", "with", "fits", "replacement", "works"}
BRAND_TERMS = {
    "adidas",
    "apple",
    "canon",
    "dyson",
    "nike",
    "samsung",
    "sony",
}
MODEL_LIKE_TERMS = {"a7r", "iii", "xr", "x100v", "r5", "r6"}
PRODUCT_CONTEXT_TERMS = {
    "camera",
    "camra",
    "charger",
    "chargerr",
    "chrger",
    "shoes",
    "shoose",
    "sneakers",
    "sneekers",
    "cleats",
}

SAFE_CORRECTIONS = {
    "chargerr": "charger",
    "chrger": "charger",
    "camra": "camera",
    "addidas": "adidas",
    "airpod": "airpods",
    "dysonn": "dyson",
    "shoose": "shoes",
    "sneekers": "sneakers",
    "lacefree": "lace-free",
    "lacless": "laceless",
}

RESULT_FIELDS = [
    "raw_query",
    "normalized_query",
    "correction_applied",
    "correction_count",
    "protected_tokens",
    "protected_token_count",
    "normalized_tokens",
    "query_noise_score",
    "normalization_confidence",
    "normalization_risk",
    "normalization_status",
    "normalization_trace",
]


@dataclass
class QueryNormalizationResult:
    raw_query: str
    normalized_query: str
    correction_applied: bool
    correction_count: int
    protected_tokens: str
    protected_token_count: int
    normalized_tokens: str
    query_noise_score: float
    normalization_confidence: float
    normalization_risk: str
    normalization_status: str
    normalization_trace: str


def clean_text(value: object) -> str:
    return str(value or "").strip()


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def tokenize(query: str) -> List[str]:
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", query.lower())


def normalize_spacing(query: str) -> Tuple[str, List[str]]:
    trace = []
    normalized = clean_text(query).lower()
    normalized = normalized.replace("_", " ")
    normalized = re.sub(r"(?i)\biphone(\d+)\b", r"iphone \1", normalized)
    if normalized != clean_text(query).lower():
        trace.append("split glued iphone model token")
    before = normalized
    normalized = re.sub(r"(?i)\b(\d+gb)(micro[-\s]?sd)\b", r"\1 micro sd", normalized)
    normalized = re.sub(r"(?i)\b(micro)-?(sd)\b", r"\1 sd", normalized)
    if normalized != before:
        trace.append("normalized micro sd spelling")
    normalized = re.sub(r"[^a-z0-9+\-.\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized, trace


def has_numeric_or_model_token(token: str) -> bool:
    if token in MODEL_LIKE_TERMS:
        return True
    if re.search(r"\d", token) and re.search(r"[a-z]", token):
        return True
    if re.fullmatch(r"\d+(gb|tb|mm|cm|in|inch|oz|ft|xl|x|w|v)?", token):
        return True
    if re.fullmatch(r"[a-z]\d+[a-z0-9]*", token):
        return True
    return False


def detect_protected_tokens(tokens: List[str], normalized: str) -> List[str]:
    protected = []
    for token in tokens:
        if (
            token in NEGATION_TERMS
            or token in COMPATIBILITY_TERMS
            or token in BRAND_TERMS
            or token in MODEL_LIKE_TERMS
            or has_numeric_or_model_token(token)
        ):
            protected.append(token)
    for phrase in ["compatible with", "replacement for", "works with"]:
        if phrase in normalized:
            protected.append(phrase)
    return list(dict.fromkeys(protected))


def compatibility_context(tokens: List[str]) -> bool:
    token_set = set(tokens)
    return bool(token_set & {"charger", "chargerr", "chrger", "camera", "camra", "case", "replacement", "filter"})


def apply_token_corrections(tokens: List[str]) -> Tuple[List[str], List[str], int, bool]:
    corrected = []
    trace = []
    correction_count = 0
    touched_protected = False
    token_set = set(tokens)
    context = compatibility_context(tokens)

    for index, token in enumerate(tokens):
        replacement = token
        reason = ""
        if token == "fr" and context:
            replacement = "for"
            reason = "corrected fr to for in compatibility context"
        elif token == "cannon" and ({"camera", "camra", "charger", "chargerr", "chrger"} & token_set):
            replacement = "canon"
            reason = "corrected cannon to canon near camera/charger context"
        elif token == "nik" and ({"shoes", "shoose", "sneakers", "sneekers", "cleats"} & token_set):
            replacement = "nike"
            reason = "corrected nik to nike near footwear context"
        elif token in SAFE_CORRECTIONS:
            replacement = SAFE_CORRECTIONS[token]
            reason = f"corrected {token} to {replacement}"

        if replacement != token:
            correction_count += 1
            trace.append(reason)
            if token in NEGATION_TERMS or token in COMPATIBILITY_TERMS or has_numeric_or_model_token(token):
                touched_protected = True
            if replacement == "lace-free":
                corrected.extend(["lace-free"])
                continue
        corrected.append(replacement)

    return corrected, trace, correction_count, touched_protected


def score_noise(raw_query: str, normalized_query: str, correction_count: int) -> float:
    raw = clean_text(raw_query)
    score = 0.0
    if re.search(r"[^a-zA-Z0-9\s+\-./]", raw):
        score += 0.15
    if re.search(r"([a-zA-Z])\1{2,}", raw):
        score += 0.20
    if correction_count:
        score += min(0.40, correction_count * 0.12)
    if re.search(r"\d+[a-zA-Z]{4,}", raw.replace(" ", "")):
        score += 0.18
    if len(raw) > 80:
        score += 0.12
    if normalized_query != raw.lower().strip():
        score += 0.05
    return round(min(score, 1.0), 4)


def risk_label(tokens: List[str], protected_tokens: List[str], touched_protected: bool) -> str:
    token_set = set(tokens)
    has_negation = bool(token_set & NEGATION_TERMS)
    has_compatibility = (
        ("for" in token_set and compatibility_context(tokens))
        or any(term in " ".join(tokens) for term in ["compatible with", "replacement for", "works with"])
    )
    dense_model = sum(1 for token in tokens if has_numeric_or_model_token(token)) >= 2
    if touched_protected or has_negation or has_compatibility or dense_model:
        return "high"
    if protected_tokens:
        return "medium"
    return "low"


def status_label(correction_count: int, protected_tokens: List[str], risk: str, touched_protected: bool) -> str:
    if risk == "high" and correction_count == 0:
        return "high_risk_preserved"
    if touched_protected:
        return "skipped_uncertain"
    if correction_count and protected_tokens:
        return "corrected_with_protected_tokens"
    if correction_count:
        return "corrected_safe"
    return "unchanged_clean"


def confidence_score(correction_count: int, noise_score: float, risk: str, touched_protected: bool) -> float:
    confidence = 0.96
    confidence -= min(0.30, correction_count * 0.04)
    confidence -= min(0.20, noise_score * 0.20)
    if risk == "medium":
        confidence -= 0.08
    if risk == "high":
        confidence -= 0.16
    if touched_protected:
        confidence -= 0.20
    return round(max(0.05, min(confidence, 0.99)), 4)


def normalize_query(query: str) -> QueryNormalizationResult:
    raw_query = clean_text(query)
    spacing_normalized, trace = normalize_spacing(raw_query)
    spacing_changed = spacing_normalized != raw_query.lower().strip()
    initial_tokens = tokenize(spacing_normalized)
    corrected_tokens, correction_trace, correction_count, touched_protected = apply_token_corrections(initial_tokens)
    trace.extend(correction_trace)
    effective_correction_count = correction_count + (1 if spacing_changed and correction_count == 0 else 0)

    normalized_query = " ".join(corrected_tokens)
    normalized_query = re.sub(r"\s+", " ", normalized_query).strip()
    normalized_tokens = tokenize(normalized_query)
    protected_tokens = detect_protected_tokens(normalized_tokens, normalized_query)

    noise_score = score_noise(raw_query, normalized_query, effective_correction_count)
    risk = risk_label(normalized_tokens, protected_tokens, touched_protected)
    status = status_label(effective_correction_count, protected_tokens, risk, touched_protected)
    confidence = confidence_score(effective_correction_count, noise_score, risk, touched_protected)
    if not trace:
        trace.append("no normalization changes applied")

    return QueryNormalizationResult(
        raw_query=raw_query,
        normalized_query=normalized_query,
        correction_applied=effective_correction_count > 0 or normalized_query != raw_query.lower().strip(),
        correction_count=effective_correction_count,
        protected_tokens="|".join(protected_tokens),
        protected_token_count=len(protected_tokens),
        normalized_tokens="|".join(normalized_tokens),
        query_noise_score=noise_score,
        normalization_confidence=confidence,
        normalization_risk=risk,
        normalization_status=status,
        normalization_trace="; ".join(trace),
    )


def summarize(rows: List[Dict[str, object]], runtime_seconds: float) -> Dict[str, object]:
    total = len(rows)
    correction_count = sum(1 for row in rows if str(row.get("correction_applied")) == "True" or row.get("correction_applied") is True)
    total_corrections = sum(int(row.get("correction_count") or 0) for row in rows)
    avg_noise = sum(float(row.get("query_noise_score") or 0.0) for row in rows) / max(total, 1)
    avg_conf = sum(float(row.get("normalization_confidence") or 0.0) for row in rows) / max(total, 1)
    high_risk_count = sum(1 for row in rows if row.get("normalization_risk") == "high")
    protected_count = sum(1 for row in rows if int(row.get("protected_token_count") or 0) > 0)
    status_counts = Counter(str(row.get("normalization_status") or "") for row in rows)
    return {
        "total_queries": total,
        "correction_applied_count": correction_count,
        "correction_rate": round(correction_count / max(total, 1), 6),
        "avg_correction_count": round(total_corrections / max(total, 1), 6),
        "avg_query_noise_score": round(avg_noise, 6),
        "avg_normalization_confidence": round(avg_conf, 6),
        "high_risk_count": high_risk_count,
        "protected_token_query_count": protected_count,
        "top_normalization_status": status_counts.most_common(1)[0][0] if status_counts else "",
        "runtime_seconds": round(runtime_seconds, 4),
    }


def report_markdown(rows: List[Dict[str, object]], summary: Dict[str, object]) -> str:
    lines = ["# MVP 24.3 Spell Check + Query Normalization Agent", "", "## Summary"]
    for key, value in summary.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Examples"])
    for row in rows[:20]:
        lines.append(
            f"- {row['raw_query']} -> {row['normalized_query']} "
            f"({row['normalization_status']}, risk={row['normalization_risk']})"
        )
    return "\n".join(lines) + "\n"


def write_outputs(rows: List[Dict[str, object]], output_dir: Path, runtime_seconds: float) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows, runtime_seconds=runtime_seconds)
    write_csv(output_dir / "query_normalization_results.csv", rows, RESULT_FIELDS)
    write_csv(output_dir / "query_normalization_summary.csv", [summary], list(summary.keys()))
    (output_dir / "query_normalization_report.md").write_text(report_markdown(rows, summary), encoding="utf-8")
    return summary


def print_single(result: QueryNormalizationResult, summary: Dict[str, object]) -> None:
    print("\nMVP 24.3 Spell Check + Query Normalization Agent")
    print("-" * 100)
    for key, value in asdict(result).items():
        print(f"{key}: {console_text(value) if isinstance(value, str) else value}")
    print("\nSummary")
    print("-" * 100)
    for key, value in summary.items():
        print(f"{key}: {value}")


def run_single(query: str, output_dir: Path) -> None:
    start = time.perf_counter()
    result = normalize_query(query)
    runtime = time.perf_counter() - start
    summary = write_outputs([asdict(result)], output_dir=output_dir, runtime_seconds=runtime)
    print_single(result, summary)


def run_batch(sample_size: int, query_mode: str, start_index: int, output_dir: Path) -> None:
    all_queries, source = load_all_queries(query_mode)
    selected = select_queries(
        queries=all_queries,
        query_mode=query_mode,
        sample_size=sample_size,
        start_index=start_index,
    )

    print("\nMVP 24.3 Spell Check + Query Normalization Agent")
    print("-" * 100)
    print(f"query_mode: {query_mode}")
    print(f"source: {source}")
    print(f"selected_query_count: {len(selected)}")

    rows = []
    start = time.perf_counter()
    for index, (_query_index, query) in enumerate(selected, start=1):
        result = normalize_query(query)
        row = asdict(result)
        rows.append(row)
        if index <= 10 or index % 100 == 0 or index == len(selected):
            print(
                f"[{index}/{len(selected)}] {console_text(result.raw_query)} -> "
                f"{console_text(result.normalized_query)} "
                f"status={result.normalization_status} risk={result.normalization_risk}"
            )

    runtime = time.perf_counter() - start
    summary = write_outputs(rows, output_dir=output_dir, runtime_seconds=runtime)

    print("\nSummary")
    print("-" * 100)
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("\nFiles written under", output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MVP 24.3 Spell Check + Query Normalization Agent")
    parser.add_argument("--query", default="", help="Single query to normalize.")
    parser.add_argument("--sample-size", type=int, default=100, help="Batch sample size.")
    parser.add_argument(
        "--query-mode",
        choices=["smoke", "esci", "stratified_esci"],
        default="smoke",
        help="Batch query source mode.",
    )
    parser.add_argument("--start-index", type=int, default=0, help="Batch start offset.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if args.query:
        run_single(args.query, output_dir=output_dir)
        return
    if args.sample_size <= 0:
        raise ValueError("--sample-size must be positive")
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative")
    run_batch(
        sample_size=args.sample_size,
        query_mode=args.query_mode,
        start_index=args.start_index,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
