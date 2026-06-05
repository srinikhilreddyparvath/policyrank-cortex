import { SearchResponse } from "../lib/types";

type SearchResultsProps = {
  response: SearchResponse | null;
};

export default function SearchResults({ response }: SearchResultsProps) {
  if (!response) {
    return null;
  }

  if (response.status !== "ok") {
    return <div className="notice error">{response.error_message || "CORTEX returned a structured error."}</div>;
  }

  return (
    <section className="resultsGrid">
      {response.top_results.map((item) => (
        <article className="resultCard" key={`${item.rank}-${item.product_id || item.title}`}>
          <div className="rank">{String(item.rank).padStart(2, "0")}</div>
          <div className="resultBody">
            <h3>{item.title || "Untitled result"}</h3>
            <div className="metaLine">
              <span>{item.brand || "Unbranded"}</span>
              {item.score !== null && <span>Score {item.score.toFixed(3)}</span>}
              {item.esci_label && <span>Label {item.esci_label}</span>}
            </div>
            <div className="badgeRow">
              {item.route_badges.filter(Boolean).slice(0, 4).map((badge) => (
                <span className="badge" key={badge}>{badge}</span>
              ))}
              {item.safety_badges.filter(Boolean).map((badge) => (
                <span className="badge safety" key={badge}>{badge}</span>
              ))}
            </div>
          </div>
        </article>
      ))}
    </section>
  );
}
