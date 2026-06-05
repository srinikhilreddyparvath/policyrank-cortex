"use client";

import { useState } from "react";
import AgentHarmony from "../components/AgentHarmony";
import MetricsPanel from "../components/MetricsPanel";
import ModeToggle from "../components/ModeToggle";
import SearchHero from "../components/SearchHero";
import SearchResults from "../components/SearchResults";
import StorySection from "../components/StorySection";
import TechnicalPanel from "../components/TechnicalPanel";
import { searchCortex } from "../lib/api";
import { SearchResponse } from "../lib/types";

export default function HomePage() {
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState<"simple" | "technical">("simple");
  const [response, setResponse] = useState<SearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function runSearch() {
    if (!query.trim()) {
      return;
    }
    setLoading(true);
    setError("");
    try {
      const result = await searchCortex({
        query: query.trim(),
        top_k: 12,
        retrieval_mode: "full_esci",
        retrieval_backend: "fts",
        index_dir: "data/esci_index_500k",
        strict_filter_mode: "hybrid",
        scale_aware_rerank_mode: "none",
        include_diagnostics: true
      });
      setResponse(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Search failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main>
      <SearchHero query={query} loading={loading} onQueryChange={setQuery} onSearch={runSearch} />
      <section className="workspace">
        <div className="workspaceHeader">
          <ModeToggle mode={mode} onChange={setMode} />
        </div>
        {error && <div className="notice error">{error}</div>}
        {response && mode === "simple" && (
          <section className="simplePanel">
            <p className="kicker">CORTEX understood</p>
            <h2>{response.diagnostics?.query_type ? String(response.diagnostics.query_type).replace(/_/g, " ") : "route-aware search"}</h2>
            <p>
              CORTEX selected <strong>{response.route || "a governed route"}</strong> and built a final slate of{" "}
              <strong>{response.final_slate_size}</strong> results from <strong>{response.execution_source || "the retrieval backend"}</strong>.
            </p>
            <p>
              {response.fallback_used
                ? `Fallback was used: ${response.fallback_reason || "the system preserved a safe route."}`
                : "Fallback was not needed. Results are ranked through the calibrated route and final slate builder."}
            </p>
          </section>
        )}
        {response && <MetricsPanel response={response} />}
        {mode === "technical" && <TechnicalPanel response={response} />}
        <SearchResults response={response} />
        <AgentHarmony response={response} />
      </section>
      <StorySection />
    </main>
  );
}
