export type ProductResult = {
  rank: number;
  product_id: string;
  title: string;
  brand: string;
  color: string;
  score: number | null;
  esci_label: string;
  retrieval_source: string;
  execution_source: string;
  route_badges: string[];
  safety_badges: string[];
  metadata: Record<string, unknown>;
};

export type SearchResponse = {
  query: string;
  route: string;
  execution_source: string;
  final_slate_size: number;
  fallback_used: boolean;
  fallback_reason: string;
  top_results: ProductResult[];
  diagnostics: Record<string, unknown>;
  runtime_seconds: number;
  status: string;
  error_message: string;
};

export type SearchPayload = {
  query: string;
  top_k: number;
  retrieval_mode: string;
  retrieval_backend: string;
  index_dir: string;
  strict_filter_mode: string;
  scale_aware_rerank_mode: string;
  include_diagnostics: boolean;
};
