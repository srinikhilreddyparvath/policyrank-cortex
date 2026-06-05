import { SearchResponse } from "../lib/types";

type MetricsPanelProps = {
  response: SearchResponse | null;
};

export default function MetricsPanel({ response }: MetricsPanelProps) {
  if (!response) {
    return null;
  }

  const metrics = [
    ["Route", response.route || "unknown"],
    ["Slate", String(response.final_slate_size)],
    ["Fallback", response.fallback_used ? "yes" : "no"],
    ["Runtime", `${response.runtime_seconds.toFixed(3)}s`]
  ];

  return (
    <div className="metrics">
      {metrics.map(([label, value]) => (
        <div className="metricCard" key={label}>
          <span>{label}</span>
          <strong>{value}</strong>
        </div>
      ))}
    </div>
  );
}
