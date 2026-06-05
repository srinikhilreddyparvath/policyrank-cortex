# CORTEX Reproducibility Runbook

This runbook reproduces the Track A paper-ready backend artifacts on Windows PowerShell.

## Environment

From the repo root:

```powershell
cd C:\Projects\policyrank-cortex
.\.venv\Scripts\Activate.ps1
```

If the virtual environment uses a different path, activate that environment before running the commands below.

## Compile Key Scripts

```powershell
python -m py_compile src/full_esci_product_index_builder.py src/full_esci_fts_index_builder.py src/full_esci_route_retrieval_evaluator.py src/full_esci_index_scale_comparison.py src/full_esci_paper_benchmark_report.py src/full_esci_1m_scale_validation_report.py src/cortex_paper_ready_package.py
```

## Build 500k Product Index

```powershell
python -m src.full_esci_product_index_builder --source full_parquet --max-rows 500000 --output-dir data/esci_index_500k --report-dir outputs/full_esci_product_index_500k
```

## Build 500k FTS

```powershell
python -m src.full_esci_fts_index_builder --index-dir data/esci_index_500k --rebuild --max-products 500000 --validate-query "wireless mouse" --validate-top-k 10
```

## Run 500k Route Evaluation

```powershell
python -m src.full_esci_route_retrieval_evaluator --sample-size 100 --query-mode indexed_queries --top-k 12 --retrieval-backend fts --index-dir data/esci_index_500k --output-dir outputs/full_esci_route_retrieval_eval_500k
```

## Run 100k vs 500k Comparison

```powershell
python -m src.full_esci_index_scale_comparison --left-name 100k --left-index-dir data/esci_index --right-name 500k --right-index-dir data/esci_index_500k --sample-size 100 --query-mode indexed_queries --top-k 12 --retrieval-backend fts --output-dir outputs/full_esci_index_scale_comparison
```

## Run Paper Benchmark Report

```powershell
python -m src.full_esci_paper_benchmark_report
```

## Build 1M Product Index

```powershell
python -m src.full_esci_product_index_builder --source full_parquet --max-rows 1000000 --output-dir data/esci_index_1m --report-dir outputs/full_esci_product_index_1m
```

## Build 1M FTS

```powershell
python -m src.full_esci_fts_index_builder --index-dir data/esci_index_1m --rebuild --max-products 1000000 --validate-query "wireless mouse" --validate-top-k 10
```

## Inspect 1M FTS

```powershell
python -m src.full_esci_fts_index_builder --index-dir data/esci_index_1m --inspect --validate-query "wireless mouse" --validate-top-k 10
```

## Run 1M Route Evaluation

```powershell
python -m src.full_esci_route_retrieval_evaluator --sample-size 100 --query-mode indexed_queries --top-k 12 --retrieval-backend fts --index-dir data/esci_index_1m --output-dir outputs/full_esci_route_retrieval_eval_1m
```

## Optional 1M Strict Boost Evaluation

```powershell
python -m src.full_esci_route_retrieval_evaluator --sample-size 100 --query-mode indexed_queries --top-k 12 --retrieval-backend fts --index-dir data/esci_index_1m --scale-aware-rerank-mode strict_boost --output-dir outputs/full_esci_route_retrieval_eval_1m_strict_boost
```

## Run 1M Validation Report

```powershell
python -m src.full_esci_1m_scale_validation_report
```

## Run Final Paper-Ready Package

```powershell
python -m src.cortex_paper_ready_package
```

## Check Git Status

```powershell
git status --short
```

Generated data and output folders are ignored by git. Do not commit generated indexes, SQLite files, or benchmark output folders unless explicitly requested.
