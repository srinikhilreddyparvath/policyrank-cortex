import { SearchPayload, SearchResponse } from "./types";

const API_BASE = process.env.NEXT_PUBLIC_CORTEX_API_URL || "http://127.0.0.1:8000";

export async function searchCortex(payload: SearchPayload): Promise<SearchResponse> {
  const response = await fetch(`${API_BASE}/api/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });

  if (!response.ok) {
    throw new Error(`CORTEX API returned ${response.status}`);
  }

  return response.json();
}
