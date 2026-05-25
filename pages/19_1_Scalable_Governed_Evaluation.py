"""
MVP 19.1: Scalable Governed Evaluation Dashboard

Interactive Streamlit dashboard for MVP 19 scalable governed evaluation.

This page reads:
- outputs/scalable_governed_eval.csv
- outputs/scalable_governed_eval_summary.csv
- outputs/scalable_governed_eval_by_route.csv
- outputs/scalable_governed_eval_by_execution_source.csv
- outputs/scalable_governed_eval_errors.csv

It shows:
- Overall scalable evaluation health
- Success rate
- Route distribution
- Execution source distribution
- Candidate coverage status
- Final slate size diagnostics
- Baseline fallback queries
- Low/no coverage queries
- Error audit
- Raw CSV inspection
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


OUTPUT_DIR = Path("outputs")

SCALABLE_EVAL_PATH = OUTPUT_DIR / "scalable_governed_eval.csv"
SCALABLE_SUMMARY_PATH = OUTPUT_DIR / "scalable_governed_eval_summary.csv"
SCALABLE_BY_ROUTE_PATH = OUTPUT_DIR / "scalable_governed_eval_by_route.csv"
SCALABLE_BY_EXECUTION_SOURCE_PATH = OUTPUT_DIR / "scalable_governed_eval_by_execution_source.csv"
SCALABLE_ERRORS_PATH = OUTPUT_DIR / "scalable_governed_eval_errors.csv"


ROUTE_COLORS = {
    "BASELINE_ONLY": "#64748b",
    "MISSION_BUILD": "#2563eb",
    "MISSION_REPAIR": "#7c3aed",
    "STRICT_REPAIR": "#d97706",
    "BEHAVIOR_AWARE_RERANK": "#059669",
    "CRITIC_REVIEW": "#dc2626",
    "REJECT_REPAIR_NARROW_QUERY": "#111827",
}


EXECUTION_SOURCE_EXPLANATIONS = {
    "behavior_aware": "Governed CORTEX executed the behavior-aware final slate.",
    "strict_repair": "Governed CORTEX executed the strict repaired slate.",
    "baseline_fallback": "Governance preserved baseline-style handling and blocked aggressive repair.",
    "fallback_no_slate_available": "Governance produced a fallback because no usable slate artifact was available.",
}


COVERAGE_EXPLANATIONS = {
    "SUPPORTED": "The query produced a usable governed final slate with reasonable coverage.",
    "GOVERNANCE_ONLY_OR_BASELINE_FALLBACK": "The query was routed to baseline/governance-only handling, usually because repair was blocked.",
    "LOW_COVERAGE": "The query produced a small final slate.",
    "LOW_SUB_INTENT_COVERAGE": "The query produced several rows but weak sub-intent diversity.",
    "NO_FINAL_SLATE": "No governed final slate was produced.",
    "ERROR": "The query failed during scalable evaluation.",
}


st.set_page_config(
    page_title="Scalable Governed Evaluation",
    page_icon="📊",
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
                radial-gradient(circle at top left, rgba(37, 99, 235, 0.12), transparent 28%),
                radial-gradient(circle at top right, rgba(124, 58, 237, 0.10), transparent 28%),
                linear-gradient(180deg, #f8fafc 0%, #ffffff 52%, #f8fafc 100%);
            color: #0f172a;
        }

        section[data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid #e2e8f0;
        }

        .hero-card {
            background: linear-gradient(135deg, #ffffff 0%, #eff6ff 45%, #f5f3ff 100%);
            border: 1px solid #bfdbfe;
            border-radius: 26px;
            padding: 30px 34px;
            margin-bottom: 20px;
            box-shadow: 0 18px 45px rgba(15, 23, 42, 0.08);
        }

        .hero-eyebrow {
            font-size: 0.82rem;
            font-weight: 850;
            color: #2563eb;
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

        .small-muted {
            color: #64748b;
            font-size: 0.92rem;
            line-height: 1.6;
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
            border: 1px solid #bfdbfe;
            border-radius: 16px;
            padding: 14px;
            color: #1e3a8a;
            font-weight: 850;
            text-align: center;
            box-shadow: 0 6px 14px rgba(37, 99, 235, 0.06);
        }

        .status-pill {
            display: inline-block;
            padding: 6px 10px;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 850;
            margin: 3px;
            background: #f8fafc;
            border: 1px solid #cbd5e1;
            color: #334155;
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


def pct(value) -> str:
    return f"{safe_float(value) * 100:.1f}%"


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


def run_scalable_eval(sample_size: int, query_mode: str, refresh: bool) -> tuple[bool, str]:
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "src.scalable_governed_evaluator",
        "--sample-size",
        str(sample_size),
        "--query-mode",
        query_mode,
    ]

    if refresh:
        cmd.append("--refresh")

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
            <div class="hero-eyebrow">MVP 19.1 · Scalable Governed Evaluation</div>
            <div class="hero-title">Proof that CORTEX works beyond hand-picked examples.</div>
            <div class="hero-subtitle">
                This dashboard summarizes scalable governed evaluation runs across multiple queries.
                It shows whether CORTEX can route, execute, preserve baseline, handle low coverage,
                and produce final governed slates across a broader query set.
            </div>
            <div class="badge-row">
                <div class="badge">Route Distribution</div>
                <div class="badge">Execution Source Mix</div>
                <div class="badge">Coverage Status</div>
                <div class="badge">Baseline Preservation</div>
                <div class="badge">Final Slate Quality</div>
                <div class="badge">Error Audit</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_flow_card() -> None:
    st.markdown(
        """
        <div class="section-card">
            <h3>Scalable evaluation flow</h3>
            <div class="small-muted">
                MVP 19 runs the governed CORTEX runner across many queries, then aggregates route,
                execution, coverage, and slate-health metrics. This tells us whether the system is working
                as an orchestrated agentic ranking layer rather than only on smoke-test examples.
            </div>
            <div class="flow-row">
                <div class="flow-step">Query Set</div>
                <div class="flow-step">Governed Runner</div>
                <div class="flow-step">Route Decision</div>
                <div class="flow-step">Final Slate</div>
                <div class="flow-step">Coverage Status</div>
                <div class="flow-step">Aggregate Metrics</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_interpretation(summary_row: dict) -> str:
    if not summary_row:
        return "Run or load a scalable evaluation to generate an interpretation."

    total = safe_int(summary_row.get("total_queries"))
    success_rate = safe_float(summary_row.get("success_rate"))
    supported = safe_int(summary_row.get("supported_queries"))
    baseline = safe_int(summary_row.get("baseline_or_governance_only_queries"))
    low_coverage = safe_int(summary_row.get("low_coverage_queries"))
    behavior_count = safe_int(summary_row.get("behavior_aware_execution_count"))
    avg_slate = safe_float(summary_row.get("avg_final_slate_size"))
    top_route = str(summary_row.get("top_governance_route", "unknown"))
    top_source = str(summary_row.get("top_execution_source", "unknown"))

    if total == 0:
        return "No scalable evaluation rows are available yet."

    return (
        f"The scalable evaluation ran on {total} queries with a success rate of {success_rate * 100:.1f}%. "
        f"{supported} queries produced supported governed slates, while {baseline} were intentionally routed to "
        f"baseline/governance-only handling. Behavior-aware execution was used for {behavior_count} queries. "
        f"The average final slate size was {avg_slate:.2f}. The most common governance route was {top_route}, "
        f"and the most common execution source was {top_source}. "
        f"{low_coverage} queries were flagged as low-coverage, which is useful for identifying where candidate coverage or fallback logic needs improvement."
    )


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
        color_discrete_sequence=["#2563eb", "#059669", "#7c3aed", "#d97706", "#dc2626", "#64748b", "#0891b2"],
    )

    fig.update_traces(textposition="outside")
    fig.update_layout(
        template="plotly_white",
        height=390,
        showlegend=False,
        margin=dict(l=20, r=20, t=65, b=90),
    )

    return fig


def plot_group_metric(group_df: pd.DataFrame, x_col: str, y_col: str, title: str):
    if group_df.empty or x_col not in group_df.columns or y_col not in group_df.columns:
        return None

    temp = group_df.copy()
    temp[y_col] = pd.to_numeric(temp[y_col], errors="coerce").fillna(0)

    fig = px.bar(
        temp,
        x=x_col,
        y=y_col,
        text=y_col,
        color=x_col,
        title=title,
        color_discrete_sequence=["#2563eb", "#059669", "#7c3aed", "#d97706", "#dc2626", "#64748b"],
    )

    fig.update_traces(textposition="outside")
    fig.update_layout(
        template="plotly_white",
        height=390,
        showlegend=False,
        margin=dict(l=20, r=20, t=65, b=90),
    )

    return fig


def plot_slate_size_distribution(eval_df: pd.DataFrame):
    if eval_df.empty or "final_slate_size" not in eval_df.columns:
        return None

    temp = eval_df.copy()
    temp["final_slate_size"] = pd.to_numeric(temp["final_slate_size"], errors="coerce").fillna(0)

    fig = px.histogram(
        temp,
        x="final_slate_size",
        nbins=12,
        title="Final Slate Size Distribution",
        color_discrete_sequence=["#2563eb"],
    )

    fig.update_layout(
        template="plotly_white",
        height=390,
        xaxis_title="Final Slate Size",
        yaxis_title="Query Count",
        margin=dict(l=20, r=20, t=65, b=40),
    )

    return fig


def prepare_eval_display(eval_df: pd.DataFrame) -> pd.DataFrame:
    if eval_df.empty:
        return eval_df

    display_cols = [
        "query",
        "success",
        "query_type",
        "governance_route",
        "governance_decision",
        "final_execution_source",
        "final_slate_size",
        "unique_sub_intents",
        "candidate_coverage_status",
        "baseline_preserved",
        "cold_start_proxy_items",
        "exploration_items",
    ]

    existing = [col for col in display_cols if col in eval_df.columns]
    return eval_df[existing].copy()


# =============================================================================
# Sidebar
# =============================================================================

with st.sidebar:
    st.markdown("## Scalable Evaluation")

    st.markdown(
        """
        <div class="small-muted">
            Run or inspect scalable governed evaluation outputs.
            Use small sample sizes for fast demos.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    sample_size = st.slider(
        "Sample size",
        min_value=5,
        max_value=100,
        value=15,
        step=5,
    )

    query_mode = st.selectbox(
        "Query mode",
        ["smoke", "mission_seed", "esci"],
        index=1,
    )

    refresh = st.checkbox(
        "Refresh full pipeline",
        value=False,
        help="Leave unchecked for fast mode. Refreshing full pipeline is slower.",
    )

    run_clicked = st.button(
        "Run Scalable Evaluation",
        type="primary",
        use_container_width=True,
    )

    st.divider()

    st.markdown("### Recommended sequence")
    st.markdown(
        """
        1. Smoke / 5 queries  
        2. Mission seed / 15 queries  
        3. ESCI / 25 queries  
        4. ESCI / 100 queries
        """
    )

    st.divider()

    st.markdown("### Backend command")
    st.code(
        '.\\.venv\\Scripts\\python.exe -u -m src.scalable_governed_evaluator --sample-size 15 --query-mode mission_seed',
        language="powershell",
    )


# =============================================================================
# Main page
# =============================================================================

render_hero()
render_flow_card()

raw_output = ""

if run_clicked:
    with st.spinner(f"Running scalable governed evaluation: {query_mode}, sample={sample_size}"):
        ok, raw_output = run_scalable_eval(
            sample_size=sample_size,
            query_mode=query_mode,
            refresh=refresh,
        )

    if ok:
        st.markdown(
            """
            <div class="success-card">
                Scalable governed evaluation completed. Output files were refreshed.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="danger-card">
                Scalable governed evaluation failed. Open the Raw Output tab to inspect the backend error.
            </div>
            """,
            unsafe_allow_html=True,
        )


eval_df = read_csv_safe(SCALABLE_EVAL_PATH)
summary_df = read_csv_safe(SCALABLE_SUMMARY_PATH)
by_route_df = read_csv_safe(SCALABLE_BY_ROUTE_PATH)
by_execution_df = read_csv_safe(SCALABLE_BY_EXECUTION_SOURCE_PATH)
errors_df = read_csv_safe(SCALABLE_ERRORS_PATH)

summary_row = summary_df.iloc[-1].to_dict() if not summary_df.empty else {}


overview_tab, routes_tab, coverage_tab, queries_tab, errors_tab, raw_tab = st.tabs(
    [
        "Overview",
        "Routes and Execution",
        "Coverage Diagnostics",
        "Query Audit",
        "Errors",
        "Raw Data",
    ]
)


with overview_tab:
    st.markdown("## Evaluation Overview")

    if not summary_row:
        st.markdown(
            """
            <div class="blue-card">
                No scalable governed evaluation summary found yet.
                Run an evaluation from the sidebar or run the backend evaluator in PowerShell.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown("### Evaluation Health")

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            metric_card(
                "Total Queries",
                safe_int(summary_row.get("total_queries")),
                "Number of queries evaluated.",
            )
        with c2:
            metric_card(
                "Success Rate",
                pct(summary_row.get("success_rate")),
                "Share of queries completed without runtime failure.",
            )
        with c3:
            metric_card(
                "Supported Queries",
                safe_int(summary_row.get("supported_queries")),
                "Queries with usable governed slates.",
            )
        with c4:
            metric_card(
                "Low Coverage",
                safe_int(summary_row.get("low_coverage_queries")),
                "Queries needing candidate/slate improvement.",
            )

        st.markdown("### Governed Execution")

        e1, e2, e3, e4 = st.columns(4)

        with e1:
            metric_card(
                "Avg Slate Size",
                f"{safe_float(summary_row.get('avg_final_slate_size')):.2f}",
                "Average governed final slate size.",
            )
        with e2:
            metric_card(
                "Avg Sub-Intents",
                f"{safe_float(summary_row.get('avg_unique_sub_intents')):.2f}",
                "Average mission coverage breadth.",
            )
        with e3:
            metric_card(
                "Behavior-Aware Executions",
                safe_int(summary_row.get("behavior_aware_execution_count")),
                "Queries using behavior-aware final slate.",
            )
        with e4:
            metric_card(
                "Baseline Preservation Rate",
                pct(summary_row.get("baseline_preservation_rate")),
                "Share routed to baseline/fallback preservation.",
            )

        st.markdown("### Result Interpretation")

        interpretation = build_interpretation(summary_row)

        st.markdown(
            f"""
            <div class="success-card">
                {interpretation}
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("Full summary row", expanded=False):
            st.dataframe(summary_df, use_container_width=True, hide_index=True)


with routes_tab:
    st.markdown("## Routes and Execution Sources")

    if eval_df.empty:
        st.info("No scalable evaluation rows found.")
    else:
        c1, c2 = st.columns(2)

        with c1:
            fig = plot_distribution(
                eval_df,
                "governance_route",
                "Governance Route Distribution",
            )
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = plot_distribution(
                eval_df,
                "final_execution_source",
                "Execution Source Distribution",
            )
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)

        c3, c4 = st.columns(2)

        with c3:
            fig = plot_group_metric(
                by_route_df,
                "governance_route",
                "avg_final_slate_size",
                "Average Slate Size by Route",
            )
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)

        with c4:
            fig = plot_group_metric(
                by_execution_df,
                "final_execution_source",
                "query_count",
                "Query Count by Execution Source",
            )
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("### By Route")
        st.dataframe(by_route_df, use_container_width=True, hide_index=True)

        st.markdown("### By Execution Source")
        st.dataframe(by_execution_df, use_container_width=True, hide_index=True)

        st.markdown("### Execution Source Glossary")

        glossary_cols = st.columns(2)
        for idx, (source, explanation) in enumerate(EXECUTION_SOURCE_EXPLANATIONS.items()):
            with glossary_cols[idx % 2]:
                st.markdown(
                    f"""
                    <div class="section-card">
                        <b>{source}</b>
                        <div class="small-muted">{explanation}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


with coverage_tab:
    st.markdown("## Coverage Diagnostics")

    if eval_df.empty:
        st.info("No coverage rows found.")
    else:
        c1, c2 = st.columns(2)

        with c1:
            fig = plot_distribution(
                eval_df,
                "candidate_coverage_status",
                "Candidate Coverage Status",
            )
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = plot_slate_size_distribution(eval_df)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Coverage Status Glossary")

        coverage_cols = st.columns(2)
        for idx, (status, explanation) in enumerate(COVERAGE_EXPLANATIONS.items()):
            with coverage_cols[idx % 2]:
                st.markdown(
                    f"""
                    <div class="section-card">
                        <b>{status}</b>
                        <div class="small-muted">{explanation}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("### Low Coverage / Fallback Queries")

        low_statuses = {
            "LOW_COVERAGE",
            "LOW_SUB_INTENT_COVERAGE",
            "NO_FINAL_SLATE",
            "GOVERNANCE_ONLY_OR_BASELINE_FALLBACK",
        }

        if "candidate_coverage_status" in eval_df.columns:
            low_df = eval_df[
                eval_df["candidate_coverage_status"].astype(str).isin(low_statuses)
            ].copy()
        else:
            low_df = pd.DataFrame()

        if low_df.empty:
            st.success("No low-coverage or fallback queries found in this run.")
        else:
            low_cols = [
                "query",
                "query_type",
                "governance_route",
                "final_execution_source",
                "final_slate_size",
                "unique_sub_intents",
                "candidate_coverage_status",
                "plain_english_reason",
            ]
            existing_low_cols = [col for col in low_cols if col in low_df.columns]
            st.dataframe(
                low_df[existing_low_cols],
                use_container_width=True,
                hide_index=True,
                height=420,
            )


with queries_tab:
    st.markdown("## Query Audit")

    if eval_df.empty:
        st.info("No query audit rows found.")
    else:
        display_df = prepare_eval_display(eval_df)

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            height=560,
        )

        with st.expander("Full scalable evaluation table", expanded=False):
            st.dataframe(eval_df, use_container_width=True, hide_index=True)

        csv_bytes = eval_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download scalable governed eval CSV",
            data=csv_bytes,
            file_name="scalable_governed_eval.csv",
            mime="text/csv",
            use_container_width=True,
        )


with errors_tab:
    st.markdown("## Error Audit")

    if errors_df.empty:
        st.markdown(
            """
            <div class="success-card">
                No runtime errors were captured in the latest scalable governed evaluation.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="warning-card">
                {len(errors_df)} queries failed during the latest scalable evaluation.
            </div>
            """,
            unsafe_allow_html=True,
        )

        error_cols = ["query", "error", "candidate_coverage_status"]
        existing_error_cols = [col for col in error_cols if col in errors_df.columns]

        st.dataframe(
            errors_df[existing_error_cols],
            use_container_width=True,
            hide_index=True,
            height=420,
        )


with raw_tab:
    st.markdown("## Raw Data and Debug Output")

    if raw_output:
        st.markdown("### Latest backend run output")
        st.code(raw_output)
    else:
        st.info("Raw backend output appears here after you run scalable evaluation from the sidebar.")

    st.markdown("### Output files")

    file_status = pd.DataFrame(
        [
            {
                "file": str(SCALABLE_EVAL_PATH),
                "exists": SCALABLE_EVAL_PATH.exists(),
            },
            {
                "file": str(SCALABLE_SUMMARY_PATH),
                "exists": SCALABLE_SUMMARY_PATH.exists(),
            },
            {
                "file": str(SCALABLE_BY_ROUTE_PATH),
                "exists": SCALABLE_BY_ROUTE_PATH.exists(),
            },
            {
                "file": str(SCALABLE_BY_EXECUTION_SOURCE_PATH),
                "exists": SCALABLE_BY_EXECUTION_SOURCE_PATH.exists(),
            },
            {
                "file": str(SCALABLE_ERRORS_PATH),
                "exists": SCALABLE_ERRORS_PATH.exists(),
            },
        ]
    )

    st.dataframe(file_status, use_container_width=True, hide_index=True)

    with st.expander("Raw summary dataframe", expanded=False):
        st.dataframe(summary_df, use_container_width=True)

    with st.expander("Raw scalable eval dataframe", expanded=False):
        st.dataframe(eval_df, use_container_width=True)

    with st.expander("Raw by-route dataframe", expanded=False):
        st.dataframe(by_route_df, use_container_width=True)

    with st.expander("Raw by-execution-source dataframe", expanded=False):
        st.dataframe(by_execution_df, use_container_width=True)

    with st.expander("Raw errors dataframe", expanded=False):
        st.dataframe(errors_df, use_container_width=True)