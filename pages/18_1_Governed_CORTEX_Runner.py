"""
MVP 18.1: Governed CORTEX Runner Streamlit Page

Interactive UI for MVP 18 End-to-End Governed CORTEX Runner.

This page:
- Runs src.governed_cortex_runner
- Shows the selected governance route
- Shows the executed governed path
- Displays the final governed slate
- Explains why baseline was preserved or why CORTEX intervened
- Shows route-level summary metrics and trace output
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


OUTPUT_DIR = Path("outputs")

GOVERNED_FINAL_SLATE_PATH = OUTPUT_DIR / "governed_cortex_final_slate.csv"
GOVERNED_SUMMARY_PATH = OUTPUT_DIR / "governed_cortex_summary.csv"
GOVERNED_TRACE_PATH = OUTPUT_DIR / "governed_cortex_trace.csv"

GOVERNANCE_DECISIONS_PATH = OUTPUT_DIR / "cortex_governance_decisions.csv"
GOVERNANCE_SUMMARY_PATH = OUTPUT_DIR / "cortex_governance_summary.csv"


QUERY_PRESETS = {
    "new apartment kitchen setup": {
        "type": "Compound setup mission",
        "expected": "Governed runner should execute behavior-aware CORTEX.",
        "why": "This query needs a balanced kitchen setup slate, not a single product-type ranking.",
    },
    "beach vacation packing list": {
        "type": "Multi-category travel mission",
        "expected": "Governed runner should execute strict repair or behavior-aware CORTEX.",
        "why": "This query needs protection, apparel, comfort, storage, and footwear coverage.",
    },
    "adidas soccer cleats": {
        "type": "Narrow branded product query",
        "expected": "Governed runner should preserve baseline-style handling.",
        "why": "This query should not be expanded into unrelated soccer or sports accessories.",
    },
    "world cup watch party": {
        "type": "Event mission query",
        "expected": "Governed runner should execute mission-aware / behavior-aware CORTEX.",
        "why": "This query likely needs food, decor, viewing, party supplies, and fan items.",
    },
    "camping trip essentials": {
        "type": "Outdoor mission query",
        "expected": "Governed runner should execute mission-aware CORTEX with guardrails.",
        "why": "This broad essentials query should preserve multiple outdoor trip needs.",
    },
}


ROUTE_COLORS = {
    "BASELINE_ONLY": "#64748b",
    "MISSION_BUILD": "#2563eb",
    "MISSION_REPAIR": "#7c3aed",
    "STRICT_REPAIR": "#d97706",
    "BEHAVIOR_AWARE_RERANK": "#059669",
    "CRITIC_REVIEW": "#dc2626",
    "REJECT_REPAIR_NARROW_QUERY": "#111827",
}


SOURCE_EXPLANATIONS = {
    "behavior_aware": "The runner executed the behavior-aware CORTEX slate. This means governance allowed CORTEX to intervene and use mission-stage scoring, behavior confidence, cold-start rescue, and policy reason codes.",
    "strict_repair": "The runner used the strict repaired slate because strict repair was selected or behavior-aware output was unavailable.",
    "baseline_fallback": "The runner preserved baseline-style handling. This usually happens for narrow branded or product-specific queries where mission repair would be risky.",
    "fallback_no_slate_available": "The runner could not find a valid repaired or behavior-aware slate, so it used the safest fallback path.",
}


st.set_page_config(
    page_title="Governed CORTEX Runner",
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
                radial-gradient(circle at top left, rgba(5, 150, 105, 0.12), transparent 28%),
                radial-gradient(circle at top right, rgba(37, 99, 235, 0.10), transparent 28%),
                linear-gradient(180deg, #f8fafc 0%, #ffffff 52%, #f8fafc 100%);
            color: #0f172a;
        }

        section[data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid #e2e8f0;
        }

        .hero-card {
            background: linear-gradient(135deg, #ffffff 0%, #ecfdf5 45%, #eff6ff 100%);
            border: 1px solid #bbf7d0;
            border-radius: 26px;
            padding: 30px 34px;
            margin-bottom: 20px;
            box-shadow: 0 18px 45px rgba(15, 23, 42, 0.08);
        }

        .hero-eyebrow {
            font-size: 0.82rem;
            font-weight: 850;
            color: #059669;
            text-transform: uppercase;
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
            max-width: 1100px;
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

        .route-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 24px;
            padding: 24px;
            box-shadow: 0 14px 34px rgba(15, 23, 42, 0.07);
            margin-bottom: 18px;
        }

        .route-label {
            font-size: 0.78rem;
            color: #64748b;
            font-weight: 850;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-bottom: 8px;
        }

        .route-value {
            font-size: 2rem;
            font-weight: 900;
            color: #0f172a;
            line-height: 1.15;
            word-break: break-word;
        }

        .small-muted {
            color: #64748b;
            font-size: 0.92rem;
            line-height: 1.6;
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

        .blue-card {
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            border-radius: 18px;
            padding: 16px 18px;
            color: #1e3a8a;
            line-height: 1.65;
            margin-bottom: 12px;
        }

        .purple-card {
            background: #f5f3ff;
            border: 1px solid #ddd6fe;
            border-radius: 18px;
            padding: 16px 18px;
            color: #4c1d95;
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
            min-height: 122px;
        }

        .metric-label {
            color: #64748b;
            font-size: 0.74rem;
            font-weight: 850;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 8px;
        }

        .metric-value {
            font-size: 1.55rem;
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
            min-width: 170px;
            background: #ffffff;
            border: 1px solid #bbf7d0;
            border-radius: 16px;
            padding: 14px;
            color: #065f46;
            font-weight: 850;
            text-align: center;
            box-shadow: 0 6px 14px rgba(5, 150, 105, 0.06);
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            flex-wrap: wrap;
        }

        .stTabs [data-baseweb="tab"] {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 999px;
            padding: 9px 18px;
            color: #334155;
            font-weight: 800;
        }

        .stTabs [aria-selected="true"] {
            background: #dcfce7;
            border-color: #86efac;
            color: #065f46;
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


def route_color(route: str) -> str:
    return ROUTE_COLORS.get(str(route), "#475569")


def run_governed_runner(query: str, fast_mode: bool = True) -> tuple[bool, str]:
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "src.governed_cortex_runner",
        "--query",
        query,
    ]

    if fast_mode:
        cmd.append("--skip-refresh")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )

    output = ""

    if result.stdout:
        output += result.stdout

    if result.stderr:
        output += "\n\nSTDERR:\n" + result.stderr

    return result.returncode == 0, output


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-eyebrow">MVP 18.1 · Governed CORTEX Runner</div>
            <div class="hero-title">The end-to-end governed execution layer.</div>
            <div class="hero-subtitle">
                Governance decides the route. The Governed CORTEX Runner executes it.
                This page shows the selected route, the actual execution path, the final governed slate,
                and the reason CORTEX either intervened or preserved baseline-style handling.
            </div>
            <div class="badge-row">
                <div class="badge">Governance Route</div>
                <div class="badge">Execution Source</div>
                <div class="badge">Final Governed Slate</div>
                <div class="badge">Baseline Preservation</div>
                <div class="badge">Behavior-Aware Execution</div>
                <div class="badge">Traceable Output</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_flow_card() -> None:
    st.markdown(
        """
        <div class="section-card">
            <h3>End-to-end governed flow</h3>
            <div class="small-muted">
                MVP 18 turns CORTEX into an orchestrated system. The Governance Agent decides the route,
                then the runner executes the selected path and writes one final governed slate.
            </div>
            <div class="flow-row">
                <div class="flow-step">User Query</div>
                <div class="flow-step">Governance Agent</div>
                <div class="flow-step">Route Decision</div>
                <div class="flow-step">Governed Runner</div>
                <div class="flow-step">Final Slate</div>
                <div class="flow-step">Trace + Explanation</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_execution_interpretation(summary_row: dict, query: str) -> str:
    if not summary_row:
        return "Run a query to generate an execution interpretation."

    route = str(summary_row.get("governance_route", "unknown"))
    decision = str(summary_row.get("governance_decision", "unknown"))
    source = str(summary_row.get("final_execution_source", "unknown"))
    slate_size = safe_int(summary_row.get("final_slate_size", 0))
    baseline_preserved = safe_int(summary_row.get("baseline_preserved", 0))
    cold_items = safe_int(summary_row.get("cold_start_proxy_items", 0))
    unique_sub_intents = safe_int(summary_row.get("unique_sub_intents", 0))

    if baseline_preserved == 1:
        return (
            f"For '{query}', governance selected {route} with decision {decision}. "
            "The runner preserved baseline-style handling and blocked aggressive mission repair. "
            "This is the correct behavior for narrow or product-specific queries."
        )

    if source == "behavior_aware":
        return (
            f"For '{query}', governance selected {route}, and the runner executed the behavior-aware CORTEX slate. "
            f"The final governed slate contains {slate_size} items across {unique_sub_intents} sub-intents, "
            f"including {cold_items} cold-start proxy/rescue items."
        )

    if source == "strict_repair":
        return (
            f"For '{query}', governance selected {route}, and the runner executed the strict repaired slate. "
            f"The final governed slate contains {slate_size} items."
        )

    return (
        f"For '{query}', governance selected {route}, and the runner used execution source {source}. "
        f"The final governed slate contains {slate_size} rows."
    )


def prepare_slate_display(slate_df: pd.DataFrame) -> pd.DataFrame:
    if slate_df.empty:
        return slate_df

    display_cols = [
        "governed_rank",
        "governed_source",
        "governance_route",
        "product_title",
        "sub_intent",
        "mission_stage",
        "behavior_confidence",
        "final_policy_score",
        "policy_reason",
        "governed_action",
    ]

    existing_cols = [col for col in display_cols if col in slate_df.columns]
    display = slate_df[existing_cols].copy()

    rename_map = {
        "governed_rank": "Rank",
        "governed_source": "Source",
        "governance_route": "Route",
        "product_title": "Product",
        "sub_intent": "Sub-Intent",
        "mission_stage": "Mission Stage",
        "behavior_confidence": "Behavior Confidence",
        "final_policy_score": "Final Policy Score",
        "policy_reason": "Policy Reason",
        "governed_action": "Governed Action",
    }

    display = display.rename(columns=rename_map)

    if "Final Policy Score" in display.columns:
        display["Final Policy Score"] = pd.to_numeric(
            display["Final Policy Score"],
            errors="coerce",
        ).round(4)

    return display


def plot_distribution(df: pd.DataFrame, column: str, title: str):
    if df.empty or column not in df.columns:
        return None

    chart_df = df[column].astype(str).value_counts().reset_index()
    chart_df.columns = [column, "count"]

    fig = px.bar(
        chart_df,
        x=column,
        y="count",
        text="count",
        color=column,
        title=title,
        color_discrete_sequence=["#059669", "#2563eb", "#7c3aed", "#d97706", "#dc2626", "#64748b"],
    )

    fig.update_traces(textposition="outside")
    fig.update_layout(
        template="plotly_white",
        height=390,
        showlegend=False,
        margin=dict(l=20, r=20, t=65, b=90),
    )

    return fig


def plot_slate_score(slate_df: pd.DataFrame):
    if slate_df.empty:
        return None

    required = {"governed_rank", "final_policy_score", "sub_intent"}
    if not required.issubset(set(slate_df.columns)):
        return None

    temp = slate_df.copy()
    temp["governed_rank"] = pd.to_numeric(temp["governed_rank"], errors="coerce")
    temp["final_policy_score"] = pd.to_numeric(temp["final_policy_score"], errors="coerce")
    temp = temp.dropna(subset=["governed_rank", "final_policy_score"])

    if temp.empty:
        return None

    fig = px.line(
        temp.sort_values("governed_rank"),
        x="governed_rank",
        y="final_policy_score",
        markers=True,
        text="sub_intent",
        title="Final Policy Score by Governed Rank",
    )

    fig.update_traces(textposition="top center")
    fig.update_layout(
        template="plotly_white",
        height=420,
        xaxis_title="Governed Rank",
        yaxis_title="Final Policy Score",
        margin=dict(l=20, r=20, t=65, b=40),
    )

    return fig


# =============================================================================
# Sidebar
# =============================================================================

with st.sidebar:
    st.markdown("## Governed CORTEX Runner")

    st.markdown(
        """
        <div class="small-muted">
            Use this page to run the end-to-end governed CORTEX path.
            The governance agent selects the route; the runner executes the route.
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

    st.markdown("### Expected Execution")
    st.success(preset["expected"])

    st.markdown("### Why this query matters")
    st.write(preset["why"])

    custom_query = st.text_input(
        "Or enter custom query",
        value=selected_query,
    )

    fast_mode = st.checkbox(
        "Fast mode: use existing governance outputs",
        value=True,
        help="Recommended for UI demos. Uncheck only when you want to refresh the full governance + behavior pipeline.",
    )

    run_clicked = st.button(
        "Run Governed CORTEX",
        type="primary",
        use_container_width=True,
    )

    st.divider()

    st.markdown("### Backend command")
    st.code(
        '.\\.venv\\Scripts\\python.exe -u -m src.governed_cortex_runner --query "beach vacation packing list" --skip-refresh',
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
        with st.spinner(f"Running Governed CORTEX for: {query}"):
            ok, raw_output = run_governed_runner(
                query=query,
                fast_mode=fast_mode,
            )

        if ok:
            st.markdown(
                """
                <div class="success-card">
                    Governed CORTEX completed successfully. Final slate, summary, and trace files were refreshed.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="danger-card">
                    Governed CORTEX failed. Open the Raw Data tab to inspect the backend error.
                </div>
                """,
                unsafe_allow_html=True,
            )


summary_df = filter_by_query(read_csv_safe(GOVERNED_SUMMARY_PATH), query)
slate_df = filter_by_query(read_csv_safe(GOVERNED_FINAL_SLATE_PATH), query)
trace_df = filter_by_query(read_csv_safe(GOVERNED_TRACE_PATH), query)

all_summary_df = read_csv_safe(GOVERNED_SUMMARY_PATH)
all_slate_df = read_csv_safe(GOVERNED_FINAL_SLATE_PATH)

if not slate_df.empty and "governed_rank" in slate_df.columns:
    slate_df["governed_rank"] = pd.to_numeric(slate_df["governed_rank"], errors="coerce")
    slate_df = slate_df.sort_values("governed_rank")

summary_row = summary_df.iloc[-1].to_dict() if not summary_df.empty else {}


overview_tab, slate_tab, charts_tab, trace_tab, raw_tab = st.tabs(
    [
        "Overview",
        "Final Governed Slate",
        "Charts",
        "Execution Trace",
        "Raw Data",
    ]
)


with overview_tab:
    st.markdown("## Governed Execution Overview")

    if not summary_row:
        st.markdown(
            """
            <div class="blue-card">
                Run a query from the sidebar to generate the governed CORTEX execution result.
                The overview will show the selected route, execution source, final slate size, and explanation.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        route = str(summary_row.get("governance_route", "unknown"))
        decision = str(summary_row.get("governance_decision", "unknown"))
        source = str(summary_row.get("final_execution_source", "unknown"))
        route_accent = route_color(route)

        st.markdown(
            f"""
            <div class="route-card" style="border-left: 9px solid {route_accent};">
                <div class="route-label">Governed Execution Route</div>
                <div class="route-value">{route}</div>
                <div style="height: 12px;"></div>
                <div class="small-muted">
                    <b>Governance decision:</b> {decision}<br>
                    <b>Execution source:</b> {source}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        source_explanation = SOURCE_EXPLANATIONS.get(source)
        if source_explanation:
            st.markdown(
                f"""
                <div class="purple-card">
                    <b>Execution meaning:</b> {source_explanation}
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("### Final Slate Metrics")

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            metric_card(
                "Final Slate Size",
                safe_int(summary_row.get("final_slate_size", 0)),
                "Rows returned by governed execution.",
            )
        with c2:
            metric_card(
                "Unique Sub-Intents",
                safe_int(summary_row.get("unique_sub_intents", 0)),
                "Mission coverage breadth.",
            )
        with c3:
            metric_card(
                "Cold-Start Proxy Items",
                safe_int(summary_row.get("cold_start_proxy_items", 0)),
                "Low-evidence useful items preserved.",
            )
        with c4:
            metric_card(
                "Baseline Preserved",
                safe_int(summary_row.get("baseline_preserved", 0)),
                "1 means aggressive CORTEX was blocked.",
            )

        st.markdown("### Governance Signals")

        s1, s2, s3, s4 = st.columns(4)

        with s1:
            metric_card(
                "Mission Likelihood",
                f"{safe_float(summary_row.get('mission_likelihood_score', 0)):.2f}",
                "Higher means mission-like query.",
            )
        with s2:
            metric_card(
                "Brand Specificity",
                f"{safe_float(summary_row.get('brand_specificity_score', 0)):.2f}",
                "Higher means branded/narrow query.",
            )
        with s3:
            metric_card(
                "Repair Risk",
                f"{safe_float(summary_row.get('repair_risk_score', 0)):.2f}",
                "Higher means repair needs caution.",
            )
        with s4:
            metric_card(
                "Critic Need",
                f"{safe_float(summary_row.get('critic_need_score', 0)):.2f}",
                "Higher means critic review is useful.",
            )

        st.markdown("### Plain-English Execution Summary")

        interpretation = build_execution_interpretation(summary_row, query)

        st.markdown(
            f"""
            <div class="success-card">
                {interpretation}
            </div>
            """,
            unsafe_allow_html=True,
        )

        backend_reason = summary_row.get("plain_english_reason", "")
        if backend_reason:
            with st.expander("Backend runner explanation", expanded=False):
                st.write(backend_reason)


with slate_tab:
    st.markdown("## Final Governed Slate")

    if slate_df.empty:
        st.markdown(
            """
            <div class="warning-card">
                No governed slate found for this query yet. Run Governed CORTEX from the sidebar.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        display_df = prepare_slate_display(slate_df)

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            height=540,
        )

        with st.expander("Advanced governed slate columns", expanded=False):
            st.dataframe(
                slate_df,
                use_container_width=True,
                hide_index=True,
            )

        csv_bytes = slate_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download governed final slate CSV",
            data=csv_bytes,
            file_name="governed_cortex_final_slate_filtered.csv",
            mime="text/csv",
            use_container_width=True,
        )


with charts_tab:
    st.markdown("## Governed CORTEX Charts")

    if summary_row:
        c1, c2 = st.columns(2)

        with c1:
            fig = plot_distribution(
                all_summary_df,
                "governance_route",
                "Governance Route Distribution",
            )
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = plot_distribution(
                all_summary_df,
                "final_execution_source",
                "Execution Source Distribution",
            )
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)

        c3, c4 = st.columns(2)

        with c3:
            fig = plot_distribution(
                all_summary_df,
                "governance_decision",
                "Governance Decision Distribution",
            )
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)

        with c4:
            fig = plot_slate_score(slate_df)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Score-by-rank chart is available only when behavior-aware scores exist.")
    else:
        st.info("Run a query to generate charts.")


with trace_tab:
    st.markdown("## Execution Trace")

    if trace_df.empty:
        st.info("No execution trace found for this query.")
    else:
        trace_cols = [
            "query",
            "governance_ok",
            "skip_refresh",
            "recommended_route",
            "governance_decision",
            "query_type",
            "final_execution_source",
            "final_slate_size",
        ]

        existing_trace_cols = [c for c in trace_cols if c in trace_df.columns]

        st.dataframe(
            trace_df[existing_trace_cols],
            use_container_width=True,
            hide_index=True,
        )

        if "governance_output_preview" in trace_df.columns:
            with st.expander("Governance output preview", expanded=False):
                st.code(str(trace_df.iloc[-1].get("governance_output_preview", "")))


with raw_tab:
    st.markdown("## Raw Data and Debug Output")

    if raw_output:
        st.markdown("### Latest backend run output")
        st.code(raw_output)
    else:
        st.info("Raw backend output appears here after you click Run Governed CORTEX.")

    st.markdown("### Output Files")

    file_status = pd.DataFrame(
        [
            {
                "file": str(GOVERNED_FINAL_SLATE_PATH),
                "exists": GOVERNED_FINAL_SLATE_PATH.exists(),
            },
            {
                "file": str(GOVERNED_SUMMARY_PATH),
                "exists": GOVERNED_SUMMARY_PATH.exists(),
            },
            {
                "file": str(GOVERNED_TRACE_PATH),
                "exists": GOVERNED_TRACE_PATH.exists(),
            },
            {
                "file": str(GOVERNANCE_DECISIONS_PATH),
                "exists": GOVERNANCE_DECISIONS_PATH.exists(),
            },
            {
                "file": str(GOVERNANCE_SUMMARY_PATH),
                "exists": GOVERNANCE_SUMMARY_PATH.exists(),
            },
        ]
    )

    st.dataframe(file_status, use_container_width=True, hide_index=True)

    with st.expander("Governed summary dataframe", expanded=False):
        st.dataframe(summary_df, use_container_width=True)

    with st.expander("Governed final slate dataframe", expanded=False):
        st.dataframe(slate_df, use_container_width=True)

    with st.expander("Governed trace dataframe", expanded=False):
        st.dataframe(trace_df, use_container_width=True)

    with st.expander("All governed summaries", expanded=False):
        st.dataframe(all_summary_df, use_container_width=True)

    with st.expander("All governed slate rows", expanded=False):
        st.dataframe(all_slate_df, use_container_width=True)