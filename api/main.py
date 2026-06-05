from __future__ import annotations

from api.schemas import HealthResponse, SearchRequest, SearchResponse
from api.service import run_cortex_search


try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(title="CORTEX Search Intelligence API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            project="CORTEX Search Intelligence",
            mvp="Track B Product Showcase",
            backend_available=True,
        )

    @app.post("/api/search", response_model=SearchResponse)
    def search(request: SearchRequest) -> SearchResponse:
        return run_cortex_search(request, diagnostics=False)

    @app.post("/api/diagnostics/search", response_model=SearchResponse)
    def diagnostics_search(request: SearchRequest) -> SearchResponse:
        return run_cortex_search(request, diagnostics=True)

except Exception:
    class _FallbackApp:
        title = "CORTEX Search Intelligence API"

    app = _FallbackApp()
