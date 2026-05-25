"""
MVP 17.1: CORTEX Governance Agent Streamlit Page

Interactive UI for MVP 17 CORTEX Governance Agent.

This page:
- Runs src.cortex_governance_agent
- Shows query type and governance route
- Displays key governance signals
- Explains allowed vs blocked modules
- Shows plain-English reasoning
- Displays governance output CSVs
- Visualizes route and decision distributions
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


OUTPUT_DIR = Path("outputs")

GOVERNANCE_DECISIONS_PATH = OUTPUT_DIR / "cortex_governance_decisions.csv"
GOVERNANCE_SUMMARY_PATH = OUTPUT_DIR / "cortex_governance_summary.csv"
GOVERNANCE_TRACE_PATH = OUTPUT_DIR / "cortex_governance_trace.csv"


QUERY_PRESETS = {
    "new apartment kitchen setup": {
        "type": "Compound mission query",
        "expected": "Mission repair + behavior-aware rerank",
        "why": "A broad setup query where CORTEX should preserve coverage and repair missing useful sub-intents.",
    },
    "beach vacation packing list": {
        "type": "Multi-category mission query",
        "expected": "Strict repair + behavior-aware rerank",
        "why": "A trip-planning query where CORTEX should accept useful repairs and reject weak ones.",
    },
    "adidas soccer cleats": {
        "type": "Narrow branded product query",
        "expected": "Reject mission repair / preserve baseline",
        "why": "A specific branded product search where CORTEX should avoid over-expanding the slate.",
    },
    "world cup watch party": {
        "type": "Event mission query",
        "expected": "Mission repair + behavior-aware rerank",
        "why": "An event-based query where CORTEX should reason across several shopping needs.",
    },
    "camping trip essentials": {
        "type": "Outdoor mission query",
        "expected": "Mission repair / critic if risky",
        "why": "A broad essentials query where coverage and repair risk matter.",
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


DECISION_EXPLANATIONS = {
    "USE_BASELINE_ONLY": "Governance recommends preserving the baseline path because the query does not need aggressive agentic repair.",
    "RUN_MISSION_REPAIR_AND_BEHAVIOR_AWARE": "Governance recommends mission repair and behavior-aware reranking because the query has broad mission intent.",
    "RUN_STRICT_REPAIR_AND_BEHAVIOR_AWARE": "Governance recommends strict repair plus behavior-aware reranking because repair is useful but needs guardrails.",
    "SEND_TO_CRITIC_REVIEW": "Governance recommends critic review because repair risk, coverage gap, or slate uncertainty is elevated.",
    "REJECT_REPAIR_USE_BASELINE": "Governance blocks mission repair because the query is narrow or product-specific.",
}


SIGNAL_EXPLANATIONS = {
    "mission_likelihood_score": "How strongly the query looks like a shopping mission instead of a single product search.",
    "compound_intent_score": "How much the query suggests multiple shopping needs or sub-intents.",
    "brand_specificity_score": "How strongly the query appears tied to a specific brand.",
    "narrow_query_score": "How likely the query is a narrow exact product search.",
    "repair_risk_score": "How risky it is to trust repair logic for this query.",
    "coverage_gap_score": "How much mission coverage appears missing or weak.",
    "behavior_rescue_signal": "How much useful low-evidence item rescue is happening.",
    "critic_need_score": "How strongly the query should be reviewed by the critic layer.",
}


st.set_page_config(
    page_title="CORTEX Governance Agent",
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
                radial-gradient(circle at top left, rgba(124, 58, 237, 0.12), transparent 28%),
                radial-gradient(circle at top right, rgba(5, 150, 105, 0.10), transparent 28%),
                linear-gradient(180deg, #f8fafc 0%, #ffffff 52%, #f8fafc 100%);
            color: #0f172a;
        }

        section[data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid #e2e8f0;
        }

        .hero-card {
            background: linear-gradient(135deg, #ffffff 0%, #f5f3ff 48%, #ecfdf5 100%);
            border: 1px solid #ddd6fe;
            border-radius: 26px;
            padding: 30px 34px;
            margin-bottom: 20px;
            box-shadow: 0 18px 45px rgba(15, 23, 42, 0.08);
        }

        .hero-eyebrow {
            font-size: 0.82rem;
            font-weight: 850;
            color: #7c3aed;
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
        }

        .metric-help {
            color: #64748b;
            font-size: 0.82rem;
            margin-top: 7px;
        }

        .module-pill {
            display: inline-block;
            padding: 7px 10px;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 850;
            margin: 4px;
        }

        .allowed-pill {
            background: #ecfdf5;
            color: #065f46;
            border: 1px solid #bbf7d0;
        }

        .blocked-pill {
            background: #fef2f2;
            color: #991b1b;
            border: 1px solid #fecaca;
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
            border: 1px solid #ddd6fe;
            border-radius: 16px;
            padding: 14px;
            color: #4c1d95;
            font-weight: 850;
            text-align: center;
            box-shadow: 0 6px 14px rgba(124, 58, 237, 0.06);
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
            background: #ede9fe;
            border-color: #c4b5fd;
            color: #4c1d95;
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


def run_governance_agent(query: str) -> tuple[bool, str]:
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "src.cortex_governance_agent",
        "--query",
        query,
    ]

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


def safe_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def safe_int(value) -> int:
    try:
        return int(float(value))
    except Exception:
        return 0


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


def parse_module_list(value) -> list[str]:
    if value is None or pd.isna(value):
        return []

    text = str(value)

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except Exception:
        pass

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except Exception:
        pass

    return [text]


def render_module_pills(modules: list[str], pill_class: str) -> None:
    if not modules:
        st.info("No modules listed.")
        return

    html = ""
    for module in modules:
        html += f'<span class="module-pill {pill_class}">{module}</span>'

    st.markdown(html, unsafe_allow_html=True)


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-eyebrow">MVP 17.1 · CORTEX Governance Agent</div>
            <div class="hero-title">The decision brain that decides when CORTEX should intervene.</div>
            <div class="hero-subtitle">
                The Governance Agent chooses whether a query should stay baseline-only, use mission repair,
                apply strict repair, activate behavior-aware reranking, or block repair for narrow branded searches.
                This prevents CORTEX from blindly running every expensive or aggressive module on every query.
            </div>
            <div class="badge-row">
                <div class="badge">Route Selection</div>
                <div class="badge">Repair Risk</div>
                <div class="badge">Coverage Gap</div>
                <div class="badge">Behavior Rescue Signal</div>
                <div class="badge">Allowed / Blocked Modules</div>
                <div class="badge">Plain-English Trace</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_flow_card() -> None:
    st.markdown(
        """
        <div class="section-card">
            <h3>Governed CORTEX flow</h3>
            <div class="small-muted">
                The Governance Agent sits above the rest of CORTEX. It looks at query structure,
                mission likelihood, brand specificity, repair risk, coverage gaps, and behavior-aware signals
                before deciding which modules are allowed to run.
            </div>
            <div class="flow-row">
                <div class="flow-step">Query Signals</div>
                <div class="flow-step">Governance Scoring</div>
                <div class="flow-step">Route Decision</div>
                <div class="flow-step">Allowed Modules</div>
                <div class="flow-step">Blocked Modules</div>
                <div class="flow-step">Governed Slate Path</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_decision_interpretation(decision_row: dict, summary_row: dict, query: str) -> str:
    if not decision_row:
        return "Run a query to generate a governance interpretation."

    route = str(decision_row.get("recommended_route", "unknown"))
    decision = str(decision_row.get("governance_decision", "unknown"))
    query_type = str(decision_row.get("query_type", "unknown"))

    mission = safe_float(decision_row.get("mission_likelihood_score", 0))
    narrow = safe_float(decision_row.get("narrow_query_score", 0))
    repair_risk = safe_float(decision_row.get("repair_risk_score", 0))
    critic = safe_float(decision_row.get("critic_need_score", 0))

    if decision == "REJECT_REPAIR_USE_BASELINE":
        return (
            f"For '{query}', the Governance Agent classified the query as {query_type}. "
            f"The narrow-query score is {narrow:.2f}, so CORTEX blocks mission repair and recommends baseline-style handling. "
            "This is important because it prevents over-expanding specific product searches."
        )

    if decision == "RUN_STRICT_REPAIR_AND_BEHAVIOR_AWARE":
        return (
            f"For '{query}', the Governance Agent found strong mission intent but non-trivial repair risk "
            f"({repair_risk:.2f}). It recommends strict repair plus behavior-aware reranking, meaning repairs are allowed "
            "only when guardrails accept them."
        )

    if decision == "SEND_TO_CRITIC_REVIEW":
        return (
            f"For '{query}', the Governance Agent found elevated critic need ({critic:.2f}). "
            "The query should go through critic review before aggressive repair is trusted."
        )

    if decision == "RUN_MISSION_REPAIR_AND_BEHAVIOR_AWARE":
        return (
            f"For '{query}', mission likelihood is high ({mission:.2f}), so CORTEX should use mission repair "
            "and behavior-aware reranking to preserve coverage, rescue useful low-evidence items, and avoid click-greedy ranking."
        )

    return (
        f"For '{query}', the Governance Agent recommends route {route} with decision {decision}. "
        "This route is selected from query structure, repair risk, coverage, and behavior-aware signals."
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
        color_discrete_sequence=["#7c3aed", "#059669", "#d97706", "#2563eb", "#dc2626", "#64748b"],
    )

    fig.update_traces(textposition="outside")
    fig.update_layout(
        template="plotly_white",
        height=390,
        showlegend=False,
        margin=dict(l=20, r=20, t=65, b=90),
    )

    return fig


def plot_signal_bar(summary_row: dict):
    if not summary_row:
        return None

    signal_keys = [
        "mission_likelihood_score",
        "compound_intent_score",
        "brand_specificity_score",
        "narrow_query_score",
        "repair_risk_score",
        "coverage_gap_score",
        "behavior_rescue_signal",
        "critic_need_score",
    ]

    data = []
    for key in signal_keys:
        data.append(
            {
                "signal": key.replace("_", " ").replace(" score", "").title(),
                "value": safe_float(summary_row.get(key, 0)),
            }
        )

    chart_df = pd.DataFrame(data)

    fig = px.bar(
        chart_df,
        x="value",
        y="signal",
        orientation="h",
        text="value",
        title="Governance Signal Scores",
        color="signal",
        color_discrete_sequence=["#7c3aed", "#059669", "#d97706", "#2563eb", "#dc2626", "#0891b2", "#f59e0b", "#64748b"],
    )

    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(
        template="plotly_white",
        height=440,
        showlegend=False,
        xaxis_title="Score",
        yaxis_title="Signal",
        margin=dict(l=20, r=40, t=65, b=40),
    )

    return fig


# =============================================================================
# Sidebar
# =============================================================================

with st.sidebar:
    st.markdown("## Governance Agent")

    st.markdown(
        """
        <div class="small-muted">
            Use this page to test the top-level CORTEX decision agent.
            It decides whether to preserve baseline, repair, rerank, or block aggressive modules.
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

    st.markdown("### Expected Behavior")
    st.success(preset["expected"])

    st.markdown("### Why this query matters")
    st.write(preset["why"])

    custom_query = st.text_input(
        "Or enter custom query",
        value=selected_query,
    )

    run_clicked = st.button(
        "Run Governance Agent",
        type="primary",
        use_container_width=True,
    )

    st.divider()

    st.markdown("### Backend command")
    st.code(
        '.\\.venv\\Scripts\\python.exe -u -m src.cortex_governance_agent --query "beach vacation packing list"',
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
        with st.spinner(f"Running CORTEX Governance Agent for: {query}"):
            ok, raw_output = run_governance_agent(query)

        if ok:
            st.markdown(
                """
                <div class="success-card">
                    Governance Agent completed successfully. Decision, summary, and trace files were refreshed.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="warning-card">
                    Governance Agent failed. Open the Raw Output tab to inspect the backend error.
                </div>
                """,
                unsafe_allow_html=True,
            )


decisions_df = filter_by_query(read_csv_safe(GOVERNANCE_DECISIONS_PATH), query)
summary_df = filter_by_query(read_csv_safe(GOVERNANCE_SUMMARY_PATH), query)
trace_df = filter_by_query(read_csv_safe(GOVERNANCE_TRACE_PATH), query)

decision_row = decisions_df.iloc[-1].to_dict() if not decisions_df.empty else {}
summary_row = summary_df.iloc[-1].to_dict() if not summary_df.empty else {}

all_decisions_df = read_csv_safe(GOVERNANCE_DECISIONS_PATH)
all_summary_df = read_csv_safe(GOVERNANCE_SUMMARY_PATH)


overview_tab, modules_tab, signals_tab, charts_tab, raw_tab = st.tabs(
    [
        "Overview",
        "Allowed / Blocked Modules",
        "Signals",
        "Charts",
        "Raw Data",
    ]
)


with overview_tab:
    st.markdown("## Governance Overview")

    if not decision_row:
        st.markdown(
            """
            <div class="blue-card">
                Run a query from the sidebar to generate the governance decision.
                The result will show the chosen route, query type, allowed modules, blocked modules, and a plain-English explanation.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        route = str(decision_row.get("recommended_route", "unknown"))
        decision = str(decision_row.get("governance_decision", "unknown"))
        query_type = str(decision_row.get("query_type", "unknown"))
        route_accent = route_color(route)

        st.markdown(
            f"""
            <div class="route-card" style="border-left: 9px solid {route_accent};">
                <div class="route-label">Recommended Route</div>
                <div class="route-value">{route}</div>
                <div style="height: 12px;"></div>
                <div class="small-muted">
                    <b>Decision:</b> {decision}<br>
                    <b>Query type:</b> {query_type}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### Key Decision Scores")

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            metric_card(
                "Mission Likelihood",
                f"{safe_float(decision_row.get('mission_likelihood_score', 0)):.2f}",
                "Higher means mission-like query.",
            )
        with c2:
            metric_card(
                "Brand Specificity",
                f"{safe_float(decision_row.get('brand_specificity_score', 0)):.2f}",
                "Higher means branded/narrow query.",
            )
        with c3:
            metric_card(
                "Repair Risk",
                f"{safe_float(decision_row.get('repair_risk_score', 0)):.2f}",
                "Higher means repair needs caution.",
            )
        with c4:
            metric_card(
                "Critic Need",
                f"{safe_float(decision_row.get('critic_need_score', 0)):.2f}",
                "Higher means critic review is useful.",
            )

        st.markdown("### Plain-English Reason")

        interpretation = build_decision_interpretation(decision_row, summary_row, query)
        st.markdown(
            f"""
            <div class="success-card">
                {interpretation}
            </div>
            """,
            unsafe_allow_html=True,
        )

        backend_reason = decision_row.get("plain_english_reason", "")
        if backend_reason:
            with st.expander("Backend governance explanation", expanded=False):
                st.write(backend_reason)

        explanation = DECISION_EXPLANATIONS.get(decision)
        if explanation:
            st.markdown(
                f"""
                <div class="purple-card">
                    <b>Decision meaning:</b> {explanation}
                </div>
                """,
                unsafe_allow_html=True,
            )


with modules_tab:
    st.markdown("## Allowed and Blocked Modules")

    if not decision_row:
        st.info("Run a query first to see module routing.")
    else:
        allowed_modules = parse_module_list(decision_row.get("allowed_modules"))
        blocked_modules = parse_module_list(decision_row.get("blocked_modules"))

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Allowed Modules")
            st.markdown(
                """
                <div class="blue-card">
                    These modules are allowed for the selected query route.
                </div>
                """,
                unsafe_allow_html=True,
            )
            render_module_pills(allowed_modules, "allowed-pill")

        with col2:
            st.markdown("### Blocked Modules")
            st.markdown(
                """
                <div class="warning-card">
                    These modules are blocked to avoid unnecessary cost, risk, or over-repair.
                </div>
                """,
                unsafe_allow_html=True,
            )
            render_module_pills(blocked_modules, "blocked-pill")

        st.markdown("### Why this matters")

        st.markdown(
            """
            <div class="section-card">
                <div class="small-muted">
                    Governance is the control layer that prevents CORTEX from overusing expensive or aggressive modules.
                    For broad mission queries, it allows mission repair and behavior-aware reranking.
                    For narrow branded queries, it blocks repair and preserves baseline-style handling.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


with signals_tab:
    st.markdown("## Governance Signals")

    if not summary_row:
        st.info("Run a query to see signal scores.")
    else:
        signal_items = [
            "mission_likelihood_score",
            "compound_intent_score",
            "brand_specificity_score",
            "narrow_query_score",
            "repair_risk_score",
            "coverage_gap_score",
            "behavior_rescue_signal",
            "critic_need_score",
        ]

        for idx in range(0, len(signal_items), 4):
            cols = st.columns(4)
            for col, signal in zip(cols, signal_items[idx:idx + 4]):
                with col:
                    metric_card(
                        signal.replace("_", " ").title(),
                        f"{safe_float(summary_row.get(signal, 0)):.2f}",
                        SIGNAL_EXPLANATIONS.get(signal, ""),
                    )

        st.markdown("### Signal Table")

        display_cols = [
            "query",
            "query_type",
            "recommended_route",
            "governance_decision",
            "mission_likelihood_score",
            "compound_intent_score",
            "brand_specificity_score",
            "narrow_query_score",
            "repair_risk_score",
            "coverage_gap_score",
            "behavior_rescue_signal",
            "critic_need_score",
            "strict_final_slate_size",
            "behavior_final_slate_size",
            "behavior_cold_start_rescue_items",
        ]

        existing_cols = [c for c in display_cols if c in summary_df.columns]

        st.dataframe(
            summary_df[existing_cols],
            use_container_width=True,
            hide_index=True,
        )


with charts_tab:
    st.markdown("## Governance Charts")

    if not decision_row:
        st.info("Run a query to generate charts.")
    else:
        c1, c2 = st.columns(2)

        with c1:
            fig = plot_signal_bar(summary_row)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)

        with c2:
            if not all_decisions_df.empty:
                fig = plot_distribution(
                    all_decisions_df,
                    "recommended_route",
                    "Governance Route Distribution",
                )
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)

        c3, c4 = st.columns(2)

        with c3:
            if not all_decisions_df.empty:
                fig = plot_distribution(
                    all_decisions_df,
                    "query_type",
                    "Query Type Distribution",
                )
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)

        with c4:
            if not all_decisions_df.empty:
                fig = plot_distribution(
                    all_decisions_df,
                    "governance_decision",
                    "Governance Decision Distribution",
                )
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)


with raw_tab:
    st.markdown("## Raw Data and Debug Output")

    if raw_output:
        st.markdown("### Latest backend run output")
        st.code(raw_output)
    else:
        st.info("Raw backend output appears here after you click Run Governance Agent.")

    st.markdown("### Output Files")

    file_status = pd.DataFrame(
        [
            {
                "file": str(GOVERNANCE_DECISIONS_PATH),
                "exists": GOVERNANCE_DECISIONS_PATH.exists(),
            },
            {
                "file": str(GOVERNANCE_SUMMARY_PATH),
                "exists": GOVERNANCE_SUMMARY_PATH.exists(),
            },
            {
                "file": str(GOVERNANCE_TRACE_PATH),
                "exists": GOVERNANCE_TRACE_PATH.exists(),
            },
        ]
    )

    st.dataframe(file_status, use_container_width=True, hide_index=True)

    with st.expander("Governance decisions dataframe", expanded=False):
        st.dataframe(decisions_df, use_container_width=True)

    with st.expander("Governance summary dataframe", expanded=False):
        st.dataframe(summary_df, use_container_width=True)

    with st.expander("Governance trace dataframe", expanded=False):
        st.dataframe(trace_df, use_container_width=True)

    with st.expander("All governance decisions", expanded=False):
        st.dataframe(all_decisions_df, use_container_width=True)

    with st.expander("All governance summaries", expanded=False):
        st.dataframe(all_summary_df, use_container_width=True)