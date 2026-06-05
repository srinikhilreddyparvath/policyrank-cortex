import { SearchResponse } from "../lib/types";

const modules = [
  ["Governance Router", ["BASELINE_ONLY", "STRICT_REPAIR", "MISSION_REPAIR", "CRITIC_REVIEW", "BEHAVIOR_AWARE_RERANK", "REJECT_REPAIR_NARROW_QUERY"]],
  ["Retrieval Backend", ["full_esci"]],
  ["Strict Repair", ["STRICT_REPAIR"]],
  ["Mission Repair", ["MISSION_REPAIR"]],
  ["Critic Review", ["CRITIC_REVIEW"]],
  ["Behavior-Aware Rerank", ["BEHAVIOR_AWARE_RERANK"]],
  ["Fallback Controller", ["fallback"]],
  ["Final Slate Builder", ["BASELINE_ONLY", "STRICT_REPAIR", "MISSION_REPAIR", "CRITIC_REVIEW", "BEHAVIOR_AWARE_RERANK", "REJECT_REPAIR_NARROW_QUERY"]]
];

type AgentHarmonyProps = {
  response: SearchResponse | null;
};

export default function AgentHarmony({ response }: AgentHarmonyProps) {
  const route = response?.route || "";
  const fallback = Boolean(response?.fallback_used);

  return (
    <section className="harmony">
      <div className="sectionTitle">
        <p>Coordinated system flow</p>
        <h2>Agent Harmony</h2>
      </div>
      <div className="moduleGrid">
        {modules.map(([name, activeRoutes]) => {
          const active = activeRoutes.includes(route) || (activeRoutes.includes("fallback") && fallback);
          return (
            <div className={`moduleCard ${active ? "online" : ""}`} key={name}>
              <span className="statusDot" />
              <strong>{name}</strong>
              <small>{active ? "active" : "standing by"}</small>
            </div>
          );
        })}
      </div>
    </section>
  );
}
