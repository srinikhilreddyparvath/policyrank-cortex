"""
MVP 16.2A: Polished Behavior-Aware CORTEX Streamlit Page

This page turns MVP 16 into a polished demo-ready UI.

It shows:
- What Behavior-Aware CORTEX does
- Query presets with clear intent types
- Behavior-aware summary cards
- Plain-English result interpretation
- Clean ranked slate
- Color-coded policy reason explanation
- Charts for policy reason, mission stage, and behavior confidence
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


OUTPUT_DIR = Path("outputs")

BEHAVIOR_SLATE_PATH = OUTPUT_DIR / "behavior_aware_cortex_slate.csv"
BEHAVIOR_SUMMARY_PATH = OUTPUT_DIR / "behavior_aware_cortex_summary.csv"
BEHAVIOR_REASONS_PATH = OUTPUT_DIR / "behavior_aware_cortex_policy_reasons.csv"


QUERY_PRESETS = {
    "new apartment kitchen setup": {
        "type": "Compound setup mission",
        "why": "Tests whether CORTEX can preserve a balanced kitchen setup slate instead of over-ranking one product type.",
    },
    "beach vacation packing list": {
        "type": "Multi-category travel mission",
        "why": "Tests whether CORTEX can balance protection, apparel, accessories, storage, and footwear.",
    },
    "adidas soccer cleats": {
        "type": "Narrow branded product query",
        "why": "Tests whether CORTEX avoids unnecessary mission repair for a specific branded product search.",
    },
    "world cup watch party": {
        "type": "Event mission query",
        "why": "Tests whether CORTEX can reason about products needed for a themed social event.",
    },
    "camping trip essentials": {
        "type": "Outdoor trip mission",
        "why": "Tests whether CORTEX can preserve multiple outdoor trip needs in one slate.",
    },
}


POLICY_COLORS = {
    "BOOST_BEHAVIOR_CORE_ITEM": "#2563eb",
    "RESCUE_COLD_START_RELEVANT": "#059669",
    "KEEP_BALANCED_MISSION_ITEM": "#7c3aed",
    "KEEP_LOW_CONFIDENCE_REVIEW": "#d97706",
    "ACCEPT_REPAIR_WITH_BEHAVIOR_GUARDRAIL": "#0891b2",
    "BOOST_MISSION_COMPLETION": "#16a34a",
    "PENALIZE_OVER_CONCENTRATION": "#dc2626",
    "KEEP_BRANDED_EXACT_MATCH": "#111827",
}


POLICY_EXPLANATIONS = {
    "BOOST_BEHAVIOR_CORE_ITEM": "Core mission item with strong behavior evidence. These are usually the anchor products for the query.",
    "RESCUE_COLD_START_RELEVANT": "Lower behavior evidence, but useful for completing the mission. CORTEX protects it from being buried.",
    "KEEP_BALANCED_MISSION_ITEM": "Useful supporting item that improves mission coverage without dominating the slate.",
    "KEEP_LOW_CONFIDENCE_REVIEW": "Weak behavior confidence. Kept for visibility, but marked for review.",
    "ACCEPT_REPAIR_WITH_BEHAVIOR_GUARDRAIL": "Repair-loop item accepted after behavior-aware review.",
    "BOOST_MISSION_COMPLETION": "Completion item that helps make the shopping mission feel complete.",
    "PENALIZE_OVER_CONCENTRATION": "Penalty applied because one sub-intent appears too often.",
    "KEEP_BRANDED_EXACT_MATCH": "Exact branded product match. Preserved without unnecessary mission expansion.",
}


MISSION_STAGE_EXPLANATIONS = {
    "core": "Primary product needed for the shopping mission.",
    "supporting": "Helpful product that supports the core mission.",
    "completion": "Finishing item that makes the mission complete.",
    "optional": "Nice-to-have item, but not central.",
    "unknown": "Role could not be confidently inferred.",
}


st.set_page_config(
    page_title="Behavior-Aware CORTEX",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# Styling
# =============================================================================

st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(59, 130, 246, 0.12), transparent 28%),
                radial-gradient(circle at top right, rgba(16, 185, 129, 0.10), transparent 30%),
                linear-gradient(180deg, #f8fafc 0%, #ffffff 52%, #f8fafc 100%);
            color: #0f172a;
        }

        section[data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid #e2e8f0;
        }

        .hero-card {
            background: linear-gradient(135deg, #ffffff 0%, #eff6ff 45%, #ecfdf5 100%);
            border: 1px solid #bfdbfe;
            border-radius: 26px;
            padding: 30px 34px;
            margin-bottom: 20px;
            box-shadow: 0 18px 45px rgba(15, 23, 42, 0.08);
        }

        .hero-eyebrow {
            font-size: 0.82rem;
            font-weight: 800;
            text-transform: uppercase;
            color: #2563eb;
            letter-spacing: 0.12em;
            margin-bottom: 8px;
        }

        .hero-title {
            font-size: 2.55rem;
            line-height: 1.08;
            font-weight: 900;
            color: #0f172a;
            letter-spacing: -0.055em;
            margin-bottom: 10px;
        }

        .hero-subtitle {
            font-size: 1.02rem;
            line-height: 1.7;
            color: #475569;
            max-width: 1080px;
        }

        .badge-row {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            margin-top: 18px;
        }

        .badge {
            background: #ffffff;
            color: #1e293b;
            border: 1px solid #cbd5e1;
            border-radius: 999px;
            padding: 8px 12px;
            font-size: 0.78rem;
            font-weight: 800;
            box-shadow: 0 4px 10px rgba(15, 23, 42, 0.04);
        }

        .section-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 22px;
            padding: 20px 22px;
            box-shadow: 0 10px 26px rgba(15, 23, 42, 0.055);
            margin-bottom: 18px;
        }

        .explain-card {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 18px;
            padding: 16px 18px;
            color: #334155;
            line-height: 1.65;
            margin-bottom: 12px;
        }

        .success-card {
            background: #ecfdf5;
            border: 1px solid #bbf7d0;
            border-radius: 18px;
            padding: 16px 18px;
            color: #065f46;
            line-height: 1.65;
            margin-bottom: 12px;
        }

        .warning-card {
            background: #fffbeb;
            border: 1px solid #fde68a;
            border-radius: 18px;
            padding: 16px 18px;
            color: #92400e;
            line-height: 1.65;
            margin-bottom: 12px;
        }

        .danger-card {
            background: #fef2f2;
            border: 1px solid #fecaca;
            border-radius: 18px;
            padding: 16px 18px;
            color: #991b1b;
            line-height: 1.65;
            margin-bottom: 12px;
        }

        .metric-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 18px;
            padding: 15px 16px;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.055);
            min-height: 104px;
        }

        .metric-label {
            color: #64748b;
            font-size: 0.78rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 8px;
        }

        .metric-value {
            font-size: 1.85rem;
            font-weight: 900;
            color: #0f172a;
            line-height: 1.1;
            word-break: break-word;
        }

        .metric-help {
            color: #64748b;
            font-size: 0.82rem;
            margin-top: 7px;
        }

        .flow-row {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            align-items: stretch;
            margin-top: 12px;
        }

        .flow-step {
            flex: 1;
            min-width: 160px;
            background: #ffffff;
            border: 1px solid #bfdbfe;
            border-radius: 16px;
            padding: 14px;
            color: #1e3a8a;
            font-weight: 850;
            text-align: center;
            box-shadow: 0 6px 14px rgba(37, 99, 235, 0.06);
        }

        .policy-pill {
            display: inline-block;
            color: white;
            padding: 5px 9px;
            border-radius: 999px;
            font-size: 0.74rem;
            font-weight: 850;
            white-space: nowrap;
        }

        .small-muted {
            color: #64748b;
            font-size: 0.92rem;
            line-height: 1.6;
        }

        .query-chip {
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            color: #1e3a8a;
            border-radius: 999px;
            padding: 7px 11px;
            display: inline-block;
            font-size: 0.82rem;
            font-weight: 800;
            margin-bottom: 8px;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }

        .stTabs [data-baseweb="tab"] {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 999px;
            padding: 9px 18px;
            color: #334155;
            font-weight: 750;
        }

        .stTabs [aria-selected="true"] {
            background: #dbeafe;
            border-color: #93c5fd;
            color: #1e3a8a;
        }

        div[data-testid="stDataFrame"] {
            border-radius: 16px;
            overflow: hidden;
        }

        h1, h2, h3 {
            color: #0f172a;
            letter-spacing: -0.025em;
        }

        .block-container {
            padding-top: 2.2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# Helpers
# =============================================================================

def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.error(f"Could not read {path}: {exc}")
        return pd.DataFrame()


def run_behavior_aware_cortex(query: str) -> tuple[bool, str]:
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "src.behavior_aware_cortex",
        "--query",
        query,
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )

    combined_output = ""

    if result.stdout:
        combined_output += result.stdout

    if result.stderr:
        combined_output += "\n\nSTDERR:\n" + result.stderr

    return result.returncode == 0, combined_output


def filter_by_query(df: pd.DataFrame, query: str) -> pd.DataFrame:
    if df.empty or "query" not in df.columns:
        return df

    return df[df["query"].astype(str).str.lower() == query.lower()].copy()


def safe_int(value) -> int:
    try:
        return int(float(value))
    except Exception:
        return 0


def safe_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def metric_card(label: str, value, help_text: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def policy_color(reason: str) -> str:
    return POLICY_COLORS.get(str(reason), "#475569")


def policy_pill(reason: str) -> str:
    color = policy_color(reason)
    return f'<span class="policy-pill" style="background:{color};">{reason}</span>'


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-eyebrow">MVP 16.2A · Behavior-Aware CORTEX</div>
            <div class="hero-title">Behavior-aware ranking without becoming click-greedy.</div>
            <div class="hero-subtitle">
                This page demonstrates how CORTEX balances behavior evidence, mission-stage coverage,
                cold-start rescue, and exploration policy. Instead of blindly boosting the most clickable-looking
                items, it protects products that help complete the user’s shopping mission.
            </div>
            <div class="badge-row">
                <div class="badge">Mission Stage Scoring</div>
                <div class="badge">Behavior Confidence</div>
                <div class="badge">Cold-Start Rescue</div>
                <div class="badge">Exploration Policy</div>
                <div class="badge">Explainable Reason Codes</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_flow_card() -> None:
    st.markdown(
        """
        <div class="section-card">
            <h3>What is happening?</h3>
            <div class="small-muted">
                CORTEX starts from the strict repaired mission slate, assigns behavior-aware evidence,
                checks whether each product plays a useful mission role, rescues relevant low-evidence items,
                and produces an explainable final policy ranking.
            </div>
            <div class="flow-row">
                <div class="flow-step">Strict Repaired Slate</div>
                <div class="flow-step">Behavior Scoring</div>
                <div class="flow-step">Mission Stage Logic</div>
                <div class="flow-step">Cold-Start Rescue</div>
                <div class="flow-step">Final Policy Rank</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_interpretation(summary_row: dict, slate_df: pd.DataFrame, query: str) -> str:
    final_size = safe_int(summary_row.get("final_behavior_slate_size", 0))
    unique_sub_intents = safe_int(summary_row.get("unique_sub_intents", 0))
    high_conf = safe_int(summary_row.get("high_confidence_items", 0))
    rescue = safe_int(summary_row.get("cold_start_rescue_items", 0))
    exploration = safe_int(summary_row.get("exploration_items", 0))

    if final_size == 0:
        return (
            f"For '{query}', CORTEX did not generate a behavior-aware slate. "
            "This is expected for narrow branded queries when no strict repaired mission slate exists. "
            "A future fallback can handle exact-match branded searches separately."
        )

    top_sub_intents = []
    if not slate_df.empty and "sub_intent" in slate_df.columns:
        top_sub_intents = (
            slate_df.sort_values("behavior_rank")
            .head(5)["sub_intent"]
            .astype(str)
            .dropna()
            .tolist()
        )

    top_text = ", ".join(top_sub_intents) if top_sub_intents else "the highest-ranked mission products"

    return (
        f"For '{query}', CORTEX produced {final_size} final ranked items across "
        f"{unique_sub_intents} unique sub-intents. It found {high_conf} high-confidence behavior-backed "
        f"items, while also rescuing {rescue} low-evidence but mission-relevant items. "
        f"The top of the slate focuses on {top_text}. "
        f"The exploration policy activated for {exploration} items, which helps avoid a purely click-greedy slate."
    )


def prepare_display_slate(slate_df: pd.DataFrame) -> pd.DataFrame:
    if slate_df.empty:
        return slate_df

    display = slate_df.copy()

    rename_map = {
        "behavior_rank": "Rank",
        "product_title": "Product",
        "sub_intent": "Sub-Intent",
        "sub_intent_role": "Role",
        "mission_stage": "Mission Stage",
        "behavior_score": "Behavior Score",
        "behavior_confidence": "Behavior Confidence",
        "final_policy_score": "Final Score",
        "policy_reason": "Policy Reason",
        "exploration_flag": "Explore",
        "cold_start_proxy": "Cold Start Proxy",
    }

    cols = [
        "behavior_rank",
        "product_title",
        "sub_intent",
        "mission_stage",
        "behavior_confidence",
        "behavior_score",
        "final_policy_score",
        "policy_reason",
        "exploration_flag",
        "cold_start_proxy",
    ]

    existing = [c for c in cols if c in display.columns]
    display = display[existing].rename(columns=rename_map)

    if "Final Score" in display.columns:
        display["Final Score"] = pd.to_numeric(display["Final Score"], errors="coerce").round(4)

    if "Behavior Score" in display.columns:
        display["Behavior Score"] = pd.to_numeric(display["Behavior Score"], errors="coerce").round(4)

    return display


def render_policy_reason_cards(reasons: list[str]) -> None:
    if not reasons:
        st.info("No policy reasons available yet.")
        return

    unique_reasons = sorted(set(str(r) for r in reasons if str(r).strip()))

    cols = st.columns(2)

    for idx, reason in enumerate(unique_reasons):
        color = policy_color(reason)
        explanation = POLICY_EXPLANATIONS.get(reason, "Policy reason generated by Behavior-Aware CORTEX.")

        with cols[idx % 2]:
            st.markdown(
                f"""
                <div class="section-card" style="border-left: 7px solid {color};">
                    {policy_pill(reason)}
                    <div style="height: 10px;"></div>
                    <div class="small-muted">{explanation}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def plot_count_chart(df: pd.DataFrame, column: str, title: str, color_sequence: list[str]):
    if df.empty or column not in df.columns:
        return None

    chart_df = df[column].astype(str).value_counts().reset_index()
    chart_df.columns = [column, "count"]

    fig = px.bar(
        chart_df,
        x=column,
        y="count",
        text="count",
        title=title,
        color=column,
        color_discrete_sequence=color_sequence,
    )

    fig.update_traces(textposition="outside")
    fig.update_layout(
        template="plotly_white",
        height=390,
        showlegend=False,
        margin=dict(l=20, r=20, t=65, b=80),
    )

    return fig


def plot_score_by_rank(slate_df: pd.DataFrame):
    required = {"behavior_rank", "final_policy_score", "sub_intent"}
    if slate_df.empty or not required.issubset(set(slate_df.columns)):
        return None

    temp = slate_df.copy()
    temp["behavior_rank"] = pd.to_numeric(temp["behavior_rank"], errors="coerce")
    temp["final_policy_score"] = pd.to_numeric(temp["final_policy_score"], errors="coerce")
    temp = temp.dropna(subset=["behavior_rank", "final_policy_score"])

    fig = px.line(
        temp.sort_values("behavior_rank"),
        x="behavior_rank",
        y="final_policy_score",
        markers=True,
        text="sub_intent",
        title="Final Policy Score by Rank",
    )

    fig.update_traces(textposition="top center")
    fig.update_layout(
        template="plotly_white",
        height=420,
        xaxis_title="Behavior-Aware Rank",
        yaxis_title="Final Policy Score",
        margin=dict(l=20, r=20, t=65, b=40),
    )

    return fig


# =============================================================================
# Sidebar
# =============================================================================

with st.sidebar:
    st.markdown("## Behavior-Aware CORTEX")

    st.markdown(
        """
        <div class="small-muted">
            Use this page to demo MVP 16 behavior-aware ranking with cold-start rescue
            and policy reason codes.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    selected_query = st.selectbox(
        "Choose a demo query",
        list(QUERY_PRESETS.keys()),
    )

    preset = QUERY_PRESETS[selected_query]

    st.markdown("### Query Type")
    st.info(preset["type"])

    st.markdown("### Why this query matters")
    st.write(preset["why"])

    custom_query = st.text_input(
        "Or enter a custom query",
        value=selected_query,
    )

    run_clicked = st.button(
        "Run Behavior-Aware CORTEX",
        type="primary",
        use_container_width=True,
    )

    st.divider()

    st.markdown("### Recommended tests")
    st.markdown(
        """
        1. `new apartment kitchen setup`
        2. `beach vacation packing list`
        3. `adidas soccer cleats`
        """
    )

    st.divider()

    st.markdown("### Backend command")
    st.code(
        '.\\.venv\\Scripts\\python.exe -u -m src.behavior_aware_cortex --query "beach vacation packing list"',
        language="powershell",
    )


query = custom_query.strip()


# =============================================================================
# Main page
# =============================================================================

render_hero()
render_flow_card()

raw_output = ""

if run_clicked:
    if not query:
        st.warning("Please enter a query.")
    else:
        with st.spinner(f"Running Behavior-Aware CORTEX for: {query}"):
            ok, raw_output = run_behavior_aware_cortex(query)

        if ok:
            st.markdown(
                """
                <div class="success-card">
                    Behavior-Aware CORTEX completed successfully. The slate, summary, and policy reason files were refreshed.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="danger-card">
                    Behavior-Aware CORTEX failed. Open the Raw Output tab to inspect the backend error.
                </div>
                """,
                unsafe_allow_html=True,
            )


summary_df = filter_by_query(read_csv_safe(BEHAVIOR_SUMMARY_PATH), query)
slate_df = filter_by_query(read_csv_safe(BEHAVIOR_SLATE_PATH), query)
reasons_df = filter_by_query(read_csv_safe(BEHAVIOR_REASONS_PATH), query)

if not slate_df.empty and "behavior_rank" in slate_df.columns:
    slate_df["behavior_rank"] = pd.to_numeric(slate_df["behavior_rank"], errors="coerce")
    slate_df = slate_df.sort_values("behavior_rank")

summary_row = summary_df.iloc[-1].to_dict() if not summary_df.empty else {}


overview_tab, slate_tab, reasons_tab, charts_tab, raw_tab = st.tabs(
    [
        "Overview",
        "Ranked Slate",
        "Policy Reasons",
        "Charts",
        "Raw Data",
    ]
)


with overview_tab:
    st.markdown("## Overview")

    if not summary_row:
        st.markdown(
            """
            <div class="explain-card">
                Run a query from the sidebar to generate a behavior-aware slate.
                The overview will explain how CORTEX balanced behavior evidence, mission coverage,
                cold-start rescue, and exploration.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="query-chip">Active Query: {query}</div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### Slate Health")

        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card(
                "Input Strict Slate",
                safe_int(summary_row.get("input_strict_slate_size", 0)),
                "Rows received from strict repair rules.",
            )
        with c2:
            metric_card(
                "Final Behavior Slate",
                safe_int(summary_row.get("final_behavior_slate_size", 0)),
                "Rows ranked by Behavior-Aware CORTEX.",
            )
        with c3:
            metric_card(
                "Unique Sub-Intents",
                safe_int(summary_row.get("unique_sub_intents", 0)),
                "Coverage breadth across the shopping mission.",
            )

        st.markdown("### Behavior Evidence")

        b1, b2, b3 = st.columns(3)
        with b1:
            metric_card(
                "High Confidence",
                safe_int(summary_row.get("high_confidence_items", 0)),
                "Items with strong behavior proxy evidence.",
            )
        with b2:
            metric_card(
                "Medium Confidence",
                safe_int(summary_row.get("medium_confidence_items", 0)),
                "Items with moderate behavior evidence.",
            )
        with b3:
            metric_card(
                "Low Confidence",
                safe_int(summary_row.get("low_confidence_items", 0)),
                "Items needing review or rescue logic.",
            )

        st.markdown("### Policy Intelligence")

        p1, p2, p3 = st.columns(3)
        with p1:
            metric_card(
                "Cold-Start Rescue",
                safe_int(summary_row.get("cold_start_rescue_items", 0)),
                "Relevant items protected despite lower behavior evidence.",
            )
        with p2:
            metric_card(
                "Exploration Items",
                safe_int(summary_row.get("exploration_items", 0)),
                "Items reserved for exploration or mission completion.",
            )
        with p3:
            metric_card(
                "Decision",
                summary_row.get("final_decision", "unknown"),
                "Backend decision status.",
            )

        interpretation = build_interpretation(summary_row, slate_df, query)

        st.markdown("### Result Interpretation")
        st.markdown(
            f"""
            <div class="success-card">
                {interpretation}
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("Backend explanation", expanded=False):
            st.write(summary_row.get("explanation", ""))


with slate_tab:
    st.markdown("## Behavior-Aware Ranked Slate")

    if slate_df.empty:
        st.markdown(
            """
            <div class="warning-card">
                No behavior-aware slate found for this query.
                For narrow branded queries, this can be expected if the strict mission repair layer did not create a slate.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        clean_slate = prepare_display_slate(slate_df)

        st.dataframe(
            clean_slate,
            use_container_width=True,
            hide_index=True,
            height=520,
        )

        with st.expander("Advanced scoring columns", expanded=False):
            advanced_cols = [
                "query",
                "behavior_rank",
                "original_strict_rank",
                "slate_source",
                "sub_intent",
                "sub_intent_role",
                "mission_stage",
                "base_relevance_proxy",
                "behavior_score",
                "behavior_confidence",
                "coverage_contribution",
                "cold_start_proxy",
                "exploration_flag",
                "over_concentration_penalty",
                "final_policy_score",
                "policy_reason",
                "product_title",
            ]

            existing_advanced_cols = [c for c in advanced_cols if c in slate_df.columns]

            st.dataframe(
                slate_df[existing_advanced_cols],
                use_container_width=True,
                hide_index=True,
            )

        csv_bytes = slate_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download behavior-aware slate CSV",
            data=csv_bytes,
            file_name="behavior_aware_cortex_slate_filtered.csv",
            mime="text/csv",
            use_container_width=True,
        )


with reasons_tab:
    st.markdown("## Policy Reason Codes")

    if slate_df.empty:
        st.info("Run a query with a generated slate to see policy reason codes.")
    else:
        reasons = slate_df["policy_reason"].astype(str).tolist() if "policy_reason" in slate_df.columns else []
        render_policy_reason_cards(reasons)

    st.markdown("### Mission Stage Glossary")

    stage_cols = st.columns(2)

    for idx, (stage, explanation) in enumerate(MISSION_STAGE_EXPLANATIONS.items()):
        with stage_cols[idx % 2]:
            st.markdown(
                f"""
                <div class="section-card">
                    <b>{stage}</b>
                    <div class="small-muted">{explanation}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("### Full Reason Table")

    if reasons_df.empty:
        st.info("No reason table found for this query.")
    else:
        reason_cols = [
            "sub_intent",
            "mission_stage",
            "behavior_score",
            "behavior_confidence",
            "policy_reason",
            "explanation",
            "product_title",
        ]

        existing_reason_cols = [c for c in reason_cols if c in reasons_df.columns]

        st.dataframe(
            reasons_df[existing_reason_cols],
            use_container_width=True,
            hide_index=True,
            height=440,
        )


with charts_tab:
    st.markdown("## Visual Diagnostics")

    if slate_df.empty:
        st.info("No chart data found for this query yet.")
    else:
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            fig = plot_count_chart(
                slate_df,
                "policy_reason",
                "Policy Reason Distribution",
                ["#2563eb", "#059669", "#7c3aed", "#d97706", "#0891b2", "#dc2626"],
            )
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)

        with chart_col2:
            fig = plot_count_chart(
                slate_df,
                "mission_stage",
                "Mission Stage Distribution",
                ["#1d4ed8", "#7c3aed", "#059669", "#d97706", "#64748b"],
            )
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)

        chart_col3, chart_col4 = st.columns(2)

        with chart_col3:
            fig = plot_count_chart(
                slate_df,
                "behavior_confidence",
                "Behavior Confidence Distribution",
                ["#059669", "#2563eb", "#d97706", "#dc2626"],
            )
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)

        with chart_col4:
            fig = plot_score_by_rank(slate_df)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)


with raw_tab:
    st.markdown("## Raw Data and Debug Output")

    if raw_output:
        st.markdown("### Latest backend run output")
        st.code(raw_output)
    else:
        st.info("Raw backend output appears here after you click Run Behavior-Aware CORTEX.")

    st.markdown("### Output files")

    file_status = pd.DataFrame(
        [
            {
                "file": str(BEHAVIOR_SLATE_PATH),
                "exists": BEHAVIOR_SLATE_PATH.exists(),
            },
            {
                "file": str(BEHAVIOR_SUMMARY_PATH),
                "exists": BEHAVIOR_SUMMARY_PATH.exists(),
            },
            {
                "file": str(BEHAVIOR_REASONS_PATH),
                "exists": BEHAVIOR_REASONS_PATH.exists(),
            },
        ]
    )

    st.dataframe(file_status, use_container_width=True, hide_index=True)

    with st.expander("Raw summary dataframe", expanded=False):
        st.dataframe(summary_df, use_container_width=True)

    with st.expander("Raw slate dataframe", expanded=False):
        st.dataframe(slate_df, use_container_width=True)

    with st.expander("Raw reasons dataframe", expanded=False):
        st.dataframe(reasons_df, use_container_width=True)