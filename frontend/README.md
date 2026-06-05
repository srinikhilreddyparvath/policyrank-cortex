# CORTEX Search Intelligence Frontend

Next.js product-showcase frontend for the CORTEX governed route-aware search backend.

## Run

```powershell
cd frontend
npm install
npm run dev
```

The frontend expects the API at:

```powershell
http://127.0.0.1:8000
```

Override with:

```powershell
$env:NEXT_PUBLIC_CORTEX_API_URL="http://127.0.0.1:8000"
npm run dev
```

## Backend

From the repo root:

```powershell
python -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

The search input is intentionally empty and open-ended. Query examples and dataset caveats are kept out of the hero search interface.
