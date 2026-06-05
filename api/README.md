# CORTEX Search Intelligence API

FastAPI wrapper for the existing CORTEX governed route-aware retrieval backend.

## Install

FastAPI is required for serving the API:

```powershell
pip install fastapi
```

`uvicorn` is already listed in the project requirements.

## Run

From the repo root:

```powershell
python -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

## Endpoints

- `GET /health`
- `POST /api/search`
- `POST /api/diagnostics/search`

The API delegates to existing CORTEX modules and does not duplicate ranking logic.
