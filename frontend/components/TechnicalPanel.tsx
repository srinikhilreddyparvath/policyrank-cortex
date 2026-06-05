import { SearchResponse } from "../lib/types";

type TechnicalPanelProps = {
  response: SearchResponse | null;
};

function text(value: unknown): string {
  if (value === null || value === undefined || value === "") {
    return "none";
  }
  return String(value);
}

export default function TechnicalPanel({ response }: TechnicalPanelProps) {
  if (!response) {
    return null;
  }

  const diagnostics = response.diagnostics || {};
  const rows = [
    ["Governance route", response.route],
    ["Execution source", response.execution_source],
    ["Query type", diagnostics.query_type],
    ["Retrieval backend", diagnostics.retrieval_backend],
    ["Strict filter", diagnostics.strict_filter_mode],
    ["Scale-aware rerank", diagnostics.scale_aware_rerank_mode],
    ["Strict clean", diagnostics.strict_clean_count],
    ["Strict removed", diagnostics.strict_removed_count],
    ["Strict demoted", diagnostics.strict_demoted_count],
    ["Runtime", `${response.runtime_seconds.toFixed(3)}s`]
  ];

  return (
    <section className="technicalPanel">
      <div className="panelHeader">
        <h2>Technical Trace</h2>
        <span>{response.status}</span>
      </div>
      <div className="techRows">
        {rows.map(([label, value]) => (
          <div className="techRow" key={label}>
            <span>{label}</span>
            <strong>{text(value)}</strong>
          </div>
        ))}
      </div>
      <pre>{text(diagnostics.adapter_trace)}</pre>
    </section>
  );
}
