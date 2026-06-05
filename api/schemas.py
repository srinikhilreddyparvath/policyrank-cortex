from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=12, ge=1, le=50)
    retrieval_mode: str = "full_esci"
    retrieval_backend: str = "fts"
    index_dir: str = "data/esci_index_500k"
    strict_filter_mode: str = "hybrid"
    scale_aware_rerank_mode: str = "none"
    include_diagnostics: bool = True


class ProductResult(BaseModel):
    rank: int
    product_id: str = ""
    title: str = ""
    brand: str = ""
    color: str = ""
    score: Optional[float] = None
    esci_label: str = ""
    retrieval_source: str = ""
    execution_source: str = ""
    route_badges: List[str] = Field(default_factory=list)
    safety_badges: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    query: str
    route: str
    execution_source: str
    final_slate_size: int
    fallback_used: bool
    fallback_reason: str = ""
    top_results: List[ProductResult] = Field(default_factory=list)
    diagnostics: Dict[str, Any] = Field(default_factory=dict)
    runtime_seconds: float
    status: str
    error_message: str = ""


class HealthResponse(BaseModel):
    status: str
    project: str
    mvp: str
    backend_available: bool
