import os
import json

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.retrieval import tfidf_search
from src.semantic_retrieval import semantic_search
from src.contracts import generate_search_contract
from src.llm_contract_agent import generate_llm_search_contract
from src.contract_filters import apply_contract_candidate_filter
from src.final_slate_enforcer import enforce_final_slate_contract
from src.experiment_logger import log_experiment_run, load_experiment_runs
from src.feedback import simulate_user_clicks, summarize_feedback
from src.logger import log_feedback, load_feedback_log, ensure_storage_exists
from src.bandit import select_policy_epsilon_greedy, summarize_policy_rewards
from src.ranking import apply_policy_rank

from src.slate_q_learning import (
    get_contract_state,
    select_q_learning_action,
    update_q_value,
    get_q_table_as_dataframe,
)

from src.policy_compiler import compile_and_apply_policy
from src.slate_reward import compute_slate_reward

from src.multi_agent_diversifier import (
    multi_agent_diversify_slate,
    compute_slate_quality_metrics,
)


st.set_page_config(
    page_title="PolicyRank-RL: CORTEX Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

ensure_storage_exists()


# =============================================================================
# Simple bright UI styling
# =============================================================================

st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(180deg, #f7fbff 0%, #f8fafc 45%, #ffffff 100%);
            color: #0f172a;
        }

        section[data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid #e2e8f0;
        }

        .main-title-card {
            background: linear-gradient(135deg, #ffffff 0%, #eef6ff 55%, #f5f3ff 100%);
            border: 1px solid #dbeafe;
            border-radius: 22px;
            padding: 28px 32px;
            margin-bottom: 18px;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.07);
        }

        .main-title {
            font-size: 2.35rem;
            font-weight: 850;
            color: #0f172a;
            letter-spacing: -0.04em;
            margin-bottom: 6px;
        }

        .main-subtitle {
            font-size: 1rem;
            color: #475569;
            line-height: 1.6;
            max-width: 1100px;
        }

        .badge-row {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            margin-top: 16px;
        }

        .badge {
            font-size: 0.78rem;
            font-weight: 700;
            padding: 7px 11px;
            border-radius: 999px;
            color: #1e293b;
            background: #ffffff;
            border: 1px solid #cbd5e1;
        }

        .soft-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 18px;
            padding: 18px 20px;
            margin-bottom: 16px;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05);
        }

        .info-card-blue {
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            border-radius: 16px;
            padding: 16px 18px;
            color: #1e3a8a;
            margin-bottom: 12px;
        }

        .info-card-green {
            background: #ecfdf5;
            border: 1px solid #bbf7d0;
            border-radius: 16px;
            padding: 16px 18px;
            color: #065f46;
            margin-bottom: 12px;
        }

        .info-card-yellow {
            background: #fffbeb;
            border: 1px solid #fde68a;
            border-radius: 16px;
            padding: 16px 18px;
            color: #92400e;
            margin-bottom: 12px;
        }

        .agent-box {
            background: #ffffff;
            border: 1px solid #dbeafe;
            border-radius: 16px;
            padding: 14px;
            text-align: center;
            font-weight: 750;
            color: #1e3a8a;
            min-height: 70px;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.06);
        }

        .agent-arrow {
            text-align: center;
            color: #2563eb;
            font-size: 1.45rem;
            font-weight: 900;
            padding-top: 18px;
        }

        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 14px 16px;
            box-shadow: 0 6px 16px rgba(15, 23, 42, 0.05);
        }

        div[data-testid="stMetricLabel"] {
            color: #64748b;
            font-weight: 650;
        }

        div[data-testid="stMetricValue"] {
            color: #0f172a;
            font-weight: 850;
        }

        .small-muted {
            color: #64748b;
            font-size: 0.92rem;
            line-height: 1.55;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }

        .stTabs [data-baseweb="tab"] {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 999px;
            padding: 8px 18px;
            color: #334155;
        }

        .stTabs [aria-selected="true"] {
            background: #dbeafe;
            border-color: #93c5fd;
            color: #1e3a8a;
        }

        h1, h2, h3 {
            color: #0f172a;
            letter-spacing: -0.02em;
        }

        .block-container {
            padding-top: 2.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# Cached loaders
# =============================================================================

@st.cache_data(show_spinner=False)
def load_products():
    return pd.read_csv("data/esci_balanced_sample.csv")


@st.cache_data(show_spinner=False)
def load_csv_if_exists(path):
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_json_if_exists(path):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def compact_cols(df, cols):
    existing = [col for col in cols if col in df.columns]
    if not existing:
        return df
    return df[existing]


def render_header():
    st.markdown(
        """
        <div class="main-title-card">
            <div class="main-title">PolicyRank-RL: CORTEX Engine</div>
            <div class="main-subtitle">
                A clean research prototype for agentic search ranking: semantic retrieval,
                LLM-generated search contracts, contract-aware filtering, slate Q-learning,
                multi-agent diversification, critic verification, and learned repair routing.
            </div>
            <div class="badge-row">
                <div class="badge">Semantic Retrieval</div>
                <div class="badge">LLM Contracts</div>
                <div class="badge">Contract Filtering</div>
                <div class="badge">Slate Q-Learning</div>
                <div class="badge">Critic Agent</div>
                <div class="badge">Learned Router</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_agent_flow():
    st.markdown("### Agent Flow")

    row1 = st.columns([1, 0.16, 1, 0.16, 1, 0.16, 1])

    with row1[0]:
        st.markdown('<div class="agent-box">Intent / Contract Agent</div>', unsafe_allow_html=True)
    with row1[1]:
        st.markdown('<div class="agent-arrow">→</div>', unsafe_allow_html=True)
    with row1[2]:
        st.markdown('<div class="agent-box">Retrieval Agent</div>', unsafe_allow_html=True)
    with row1[3]:
        st.markdown('<div class="agent-arrow">→</div>', unsafe_allow_html=True)
    with row1[4]:
        st.markdown('<div class="agent-box">Filtering Agent</div>', unsafe_allow_html=True)
    with row1[5]:
        st.markdown('<div class="agent-arrow">→</div>', unsafe_allow_html=True)
    with row1[6]:
        st.markdown('<div class="agent-box">Ranking Policy Agent</div>', unsafe_allow_html=True)

    row2 = st.columns([1, 0.16, 1, 0.16, 1, 0.16, 1])

    with row2[0]:
        st.markdown('<div class="agent-box">Diversifier</div>', unsafe_allow_html=True)
    with row2[1]:
        st.markdown('<div class="agent-arrow">→</div>', unsafe_allow_html=True)
    with row2[2]:
        st.markdown('<div class="agent-box">Slate Enforcer</div>', unsafe_allow_html=True)
    with row2[3]:
        st.markdown('<div class="agent-arrow">→</div>', unsafe_allow_html=True)
    with row2[4]:
        st.markdown('<div class="agent-box">Critic / Verifier</div>', unsafe_allow_html=True)
    with row2[5]:
        st.markdown('<div class="agent-arrow">→</div>', unsafe_allow_html=True)
    with row2[6]:
        st.markdown('<div class="agent-box">Repair Router</div>', unsafe_allow_html=True)


# =============================================================================
# Contract helper functions
# =============================================================================

def build_contract(
    query_text,
    selected_policy_text,
    retrieved_df,
    contract_generation_mode,
):
    if contract_generation_mode == "LLM Agent":
        sample_products = []

        if retrieved_df is not None and len(retrieved_df) > 0:
            sample_products = retrieved_df.head(8).to_dict(orient="records")

        return generate_llm_search_contract(
            query=query_text,
            sample_products=sample_products,
        )

    return generate_search_contract(query_text, selected_policy_text)


def show_contract_metrics(contract):
    source = contract.get("contract_source", "rule_based")
    fallback_used = contract.get("fallback_used", False)
    intent = contract.get("intent", {})

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Contract Source", source)
    c2.metric("Fallback Used", str(fallback_used))
    c3.metric("Product Type", intent.get("product_type", "unknown"))
    c4.metric("Price Sensitivity", intent.get("price_sensitivity", "unknown"))
    c5.metric("Quality Preference", intent.get("quality_preference", "unknown"))

    if "ranking_weights" in contract:
        with st.expander("View LLM Ranking Weights", expanded=False):
            st.json(contract.get("ranking_weights", {}))

    if "dynamic_filters" in contract:
        with st.expander("View Dynamic LLM Contract Filters", expanded=False):
            st.json(contract.get("dynamic_filters", {}))


# =============================================================================
# Charts
# =============================================================================

def reward_comparison_chart(summary_df):
    if len(summary_df) == 0:
        return None

    row = summary_df.iloc[0]

    chart_df = pd.DataFrame(
        {
            "Strategy": [
                "Baseline",
                "Gated CORTEX",
                "Full CORTEX",
                "Router CORTEX",
                "Oracle",
            ],
            "Reward@5": [
                safe_float(row.get("avg_baseline_reward_at_5")),
                safe_float(row.get("avg_gated_cortex_reward_at_5")),
                safe_float(row.get("avg_full_cortex_reward_at_5")),
                safe_float(row.get("avg_router_reward_at_5")),
                safe_float(row.get("avg_oracle_reward_at_5")),
            ],
        }
    )

    fig = px.bar(
        chart_df,
        x="Strategy",
        y="Reward@5",
        text="Reward@5",
        title="Reward@5 Comparison",
        color="Strategy",
        color_discrete_sequence=[
            "#94a3b8",
            "#3b82f6",
            "#8b5cf6",
            "#10b981",
            "#f59e0b",
        ],
    )

    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig.update_layout(
        template="plotly_white",
        height=420,
        showlegend=False,
        margin=dict(l=20, r=20, t=70, b=40),
    )

    return fig


def router_policy_distribution_chart(summary_df):
    if len(summary_df) == 0:
        return None

    row = summary_df.iloc[0]

    chart_df = pd.DataFrame(
        {
            "Policy": ["Baseline", "Gated CORTEX", "Full CORTEX"],
            "Router Selected": [
                int(row.get("predicted_baseline_count", 0)),
                int(row.get("predicted_gated_cortex_count", 0)),
                int(row.get("predicted_full_cortex_count", 0)),
            ],
            "Oracle Best": [
                int(row.get("oracle_baseline_count", 0)),
                int(row.get("oracle_gated_cortex_count", 0)),
                int(row.get("oracle_full_cortex_count", 0)),
            ],
        }
    )

    melted = chart_df.melt(
        id_vars="Policy",
        value_vars=["Router Selected", "Oracle Best"],
        var_name="Type",
        value_name="Query Count",
    )

    fig = px.bar(
        melted,
        x="Policy",
        y="Query Count",
        color="Type",
        barmode="group",
        text="Query Count",
        title="Router Selection vs Oracle Best",
        color_discrete_sequence=["#2563eb", "#f59e0b"],
    )

    fig.update_layout(
        template="plotly_white",
        height=420,
        margin=dict(l=20, r=20, t=70, b=40),
    )

    return fig


def route_reward_chart(by_route_df):
    if len(by_route_df) == 0:
        return None

    cols = [
        "governance_route",
        "avg_gated_cortex_reward_at_5",
        "avg_full_cortex_reward_at_5",
        "avg_router_reward_at_5",
        "avg_oracle_reward_at_5",
    ]

    existing = [col for col in cols if col in by_route_df.columns]

    if len(existing) < 2:
        return None

    temp = by_route_df[existing].copy()

    rename_map = {
        "avg_gated_cortex_reward_at_5": "Gated CORTEX",
        "avg_full_cortex_reward_at_5": "Full CORTEX",
        "avg_router_reward_at_5": "Router CORTEX",
        "avg_oracle_reward_at_5": "Oracle",
    }

    temp = temp.rename(columns=rename_map)

    melted = temp.melt(
        id_vars="governance_route",
        var_name="Strategy",
        value_name="Reward@5",
    )

    fig = px.bar(
        melted,
        x="governance_route",
        y="Reward@5",
        color="Strategy",
        barmode="group",
        title="Reward@5 by Governance Route",
        color_discrete_sequence=["#3b82f6", "#8b5cf6", "#10b981", "#f59e0b"],
    )

    fig.update_layout(
        template="plotly_white",
        height=450,
        margin=dict(l=20, r=20, t=70, b=40),
        xaxis_title="Governance Route",
    )

    return fig


def router_3d_chart(by_policy_df):
    if len(by_policy_df) == 0:
        return None

    required = [
        "router_predicted_policy",
        "avg_router_reward_at_5",
        "avg_router_delta_vs_gated",
        "query_count",
    ]

    if not all(col in by_policy_df.columns for col in required):
        return None

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=by_policy_df["avg_router_reward_at_5"],
                y=by_policy_df["avg_router_delta_vs_gated"],
                z=by_policy_df["query_count"],
                mode="markers+text",
                text=by_policy_df["router_predicted_policy"],
                textposition="top center",
                marker=dict(
                    size=10,
                    opacity=0.9,
                    color=by_policy_df["query_count"],
                    colorscale="Blues",
                    showscale=True,
                    colorbar=dict(title="Queries"),
                ),
            )
        ]
    )

    fig.update_layout(
        title="Router Policy Landscape",
        template="plotly_white",
        height=480,
        scene=dict(
            xaxis_title="Router Reward@5",
            yaxis_title="Delta vs Gated",
            zaxis_title="Query Count",
        ),
        margin=dict(l=0, r=0, t=60, b=0),
    )

    return fig


# =============================================================================
# Live Search Console
# =============================================================================

def render_search_console(products):
    st.markdown("## Live Search Console")

    st.markdown(
        """
        <div class="small-muted">
            Run a query through the working CORTEX pipeline: retrieval, contract generation,
            contract filtering, policy ranking, final slate enforcement, optional Q-learning,
            and multi-agent diversification.
        </div>
        """,
        unsafe_allow_html=True,
    )

    top_controls = st.columns([2.2, 1, 1, 1.4, 1])

    with top_controls[0]:
        query = st.text_input(
            "Enter a shopping search query",
            placeholder="example: adidas soccer cleats",
        )

    with top_controls[1]:
        retrieval_mode = st.selectbox(
            "Retrieval mode",
            ["Semantic", "TF-IDF"],
        )

    with top_controls[2]:
        contract_mode = st.radio(
            "Contract mode",
            [
                "LLM Agent",
                "Rule-based",
            ],
        )

    with top_controls[3]:
        policy_mode = st.radio(
            "Policy mode",
            [
                "Slate Q-Learning + Multi-Agent Diversification",
                "Slate Q-Learning",
                "Bandit auto-select",
                "Manual",
            ],
        )

    with top_controls[4]:
        manual_preference = st.selectbox(
            "Manual objective",
            ["Most relevant", "Best rating", "Lowest price"],
        )

    st.divider()

    if not query:
        st.markdown(
            """
            <div class="info-card-blue">
                Type a search query above to start. Recommended setting:
                <b>Semantic + LLM Agent + Slate Q-Learning + Multi-Agent Diversification</b>.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    st.markdown(f"### Query: `{query}`")

    with st.spinner("Running CORTEX pipeline..."):
        if retrieval_mode == "TF-IDF":
            baseline_results = tfidf_search(query, products, top_k=20)
            score_column = "retrieval_score"
            baseline_title = "Baseline TF-IDF Ranking"
        else:
            baseline_results = semantic_search(query, products, top_k=20)
            score_column = "semantic_score"
            baseline_title = "Baseline Semantic Ranking"

        if len(baseline_results) > 0 and "baseline_score" in baseline_results.columns:
            retrieval_confidence = float(baseline_results["baseline_score"].max())
        else:
            retrieval_confidence = 0.0

        selected_rl_action = None
        selected_action_weights = None
        contract_state = None
        agent_trace = None
        slate_quality = None
        enforcement_report = None

        feedback_log = load_feedback_log()

        if policy_mode == "Bandit auto-select":
            selected_policy = select_policy_epsilon_greedy(feedback_log, epsilon=0.2)

        elif policy_mode in [
            "Slate Q-Learning",
            "Slate Q-Learning + Multi-Agent Diversification",
        ]:
            selected_policy = "RL-optimized slate policy"

        else:
            selected_policy = manual_preference

        contract = build_contract(
            query_text=query,
            selected_policy_text=selected_policy,
            retrieved_df=baseline_results,
            contract_generation_mode=contract_mode,
        )

        contract_filtered_results = apply_contract_candidate_filter(
            df=baseline_results,
            contract=contract,
            min_match_score=0.50,
            strict_top_k=20,
        )

        if contract_filtered_results is not None and len(contract_filtered_results) > 0:
            ranking_input_results = contract_filtered_results
        else:
            ranking_input_results = baseline_results

        if policy_mode == "Bandit auto-select":
            policy_results = apply_policy_rank(ranking_input_results, selected_policy)

        elif policy_mode in [
            "Slate Q-Learning",
            "Slate Q-Learning + Multi-Agent Diversification",
        ]:
            contract_state = get_contract_state(
                contract=contract,
                retrieval_confidence=retrieval_confidence,
            )

            selected_rl_action, selected_action_weights = select_q_learning_action(
                state=contract_state,
                epsilon=0.25,
            )

            policy_results = compile_and_apply_policy(
                df=ranking_input_results,
                action_weights=selected_action_weights,
                retrieval_mode=retrieval_mode,
            )

            selected_policy = selected_rl_action

        else:
            policy_results = apply_policy_rank(ranking_input_results, selected_policy)

        contract_enforced_results, enforcement_report = enforce_final_slate_contract(
            df=policy_results,
            contract=contract,
            query=query,
            top_k=20,
        )

        if len(baseline_results) > 0:
            if policy_mode == "Slate Q-Learning + Multi-Agent Diversification":
                diversified_results, agent_trace = multi_agent_diversify_slate(
                    ranked_df=contract_enforced_results,
                    top_k=10,
                )

                feedback_results = simulate_user_clicks(diversified_results)
                slate_quality = compute_slate_quality_metrics(feedback_results, top_k=5)

            else:
                feedback_results = simulate_user_clicks(contract_enforced_results)

            slate_reward = compute_slate_reward(feedback_results, top_k=5)
        else:
            feedback_results = pd.DataFrame()
            slate_reward = 0.0

    if len(baseline_results) == 0:
        st.warning("No matching products found.")
        return

    st.markdown("### Run Summary")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Selected Policy", selected_policy)
    c2.metric("Retrieval Confidence", round(retrieval_confidence, 4))
    c3.metric("Retrieved Candidates", len(baseline_results))
    c4.metric("Filtered Candidates", len(ranking_input_results))
    c5.metric("Slate Reward@5", round(slate_reward, 4))

    st.markdown("### Agentic Search Contract")
    show_contract_metrics(contract)

    with st.expander("View Full Generated Search Contract", expanded=False):
        st.json(contract)

    st.markdown("### Contract-Aware Candidate Filtering")

    f_col1, f_col2, f_col3 = st.columns(3)

    f_col1.metric("Original Retrieved Candidates", len(baseline_results))
    f_col2.metric("After Contract Filter", len(ranking_input_results))

    if "contract_match_score" in ranking_input_results.columns:
        avg_contract_score = float(ranking_input_results["contract_match_score"].mean())
    else:
        avg_contract_score = 0.0

    f_col3.metric("Avg Contract Match Score", round(avg_contract_score, 4))

    with st.expander("View Contract-Filtered Candidate Pool", expanded=False):
        filter_cols = [
            "product_id",
            "product_title",
            "contract_match_score",
            "contract_filter_reason",
            "query",
            "category",
            "brand",
            "price",
            "rating",
            "esci_label",
            score_column,
            "baseline_score",
        ]

        st.dataframe(
            compact_cols(ranking_input_results, filter_cols),
            use_container_width=True,
        )

    if enforcement_report:
        st.markdown("### Final Slate Contract Enforcement")

        e_col1, e_col2, e_col3, e_col4, e_col5 = st.columns(5)

        e_col1.metric("Enforcement Status", enforcement_report.get("enforcement_status", "unknown"))
        e_col2.metric("Preferred Brand", enforcement_report.get("preferred_brand", "none"))
        e_col3.metric("Positive Rows", enforcement_report.get("positive_contract_rows", 0))
        e_col4.metric("Blocked Rows", enforcement_report.get("blocked_rows", 0))
        e_col5.metric("Low Coverage", str(enforcement_report.get("low_coverage", False)))

        if enforcement_report.get("low_coverage", False):
            st.markdown(
                f"""
                <div class="info-card-yellow">
                    {enforcement_report.get("warning", "Low coverage warning.")}
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="info-card-green">
                    {enforcement_report.get("warning", "Contract coverage looks healthy.")}
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.expander("View Final Contract-Enforced Candidate Pool", expanded=False):
            enforcement_cols = [
                "product_id",
                "product_title",
                "contract_match_score",
                "contract_filter_reason",
                "is_blocked_by_contract",
                "brand_preference_score",
                "final_contract_score",
                "final_slate_enforcement_reason",
                "query",
                "category",
                "brand",
                "esci_label",
                score_column,
                "baseline_score",
            ]

            st.dataframe(
                compact_cols(contract_enforced_results, enforcement_cols),
                use_container_width=True,
            )

    if policy_mode in [
        "Slate Q-Learning",
        "Slate Q-Learning + Multi-Agent Diversification",
    ]:
        st.markdown("### RL Decision State")

        rl_col1, rl_col2, rl_col3 = st.columns(3)

        rl_col1.metric("Contract State", contract_state)
        rl_col2.metric("Selected RL Action", selected_rl_action)
        rl_col3.metric("Retrieval Confidence", round(retrieval_confidence, 4))

        with st.expander("View Selected RL Action Weights", expanded=False):
            st.json(selected_action_weights)

    action_col1, action_col2 = st.columns(2)

    with action_col1:
        if st.button("Save feedback and update learning", use_container_width=True):
            log_feedback(query, selected_policy, feedback_results)

            if policy_mode in [
                "Slate Q-Learning",
                "Slate Q-Learning + Multi-Agent Diversification",
            ]:
                update_q_value(
                    state=contract_state,
                    action=selected_rl_action,
                    reward=slate_reward,
                    next_state=contract_state,
                )

                st.success(
                    f"Feedback saved and Q-table updated. Slate reward: {round(slate_reward, 4)}"
                )
            else:
                st.success("Feedback saved to storage/feedback_log.csv")

    with action_col2:
        if st.button("Save experiment snapshot", use_container_width=True):
            log_experiment_run(
                query=query,
                retrieval_mode=retrieval_mode,
                contract_mode=contract_mode,
                policy_mode=policy_mode,
                selected_policy=selected_policy,
                contract=contract,
                enforcement_report=enforcement_report,
                final_results=feedback_results,
                slate_reward=slate_reward,
            )

            st.success("Experiment snapshot saved to storage/experiment_runs.csv")

    st.markdown("### Slate Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"#### {baseline_title}")

        baseline_display_cols = [
            "product_id",
            "product_title",
            "query",
            "category",
            "brand",
            "price",
            "rating",
            "esci_label",
            score_column,
            "baseline_score",
        ]

        st.dataframe(
            compact_cols(baseline_results, baseline_display_cols),
            use_container_width=True,
            height=430,
        )

    with col2:
        st.markdown("#### PolicyRank / CORTEX-Reranked Slate")

        preferred_cols = [
            "position_agent",
            "product_id",
            "product_title",
            "contract_match_score",
            "contract_filter_reason",
            "is_blocked_by_contract",
            "brand_preference_score",
            "final_contract_score",
            "final_slate_enforcement_reason",
            "contract_priority_score",
            "query",
            "category",
            "brand",
            "price",
            "rating",
            "esci_label",
            score_column,
            "baseline_score",
            "esci_score",
            "rating_score",
            "diversity_score",
            "diversity_score_against_selected",
            "agent_score",
            "policy_rank_score",
            "simulated_click",
            "simulated_reward",
        ]

        st.dataframe(
            compact_cols(feedback_results, preferred_cols),
            use_container_width=True,
            height=430,
        )

    if agent_trace is not None:
        with st.expander("Multi-Agent Position Selection Trace", expanded=False):
            st.dataframe(agent_trace, use_container_width=True)

    st.markdown("### Slate Reward Summary")

    summary = summarize_feedback(feedback_results)

    metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)

    metric_col1.metric("Total Results", summary["total_results"])
    metric_col2.metric("Simulated Clicks", summary["total_clicks"])
    metric_col3.metric("Total Reward", round(summary["total_reward"], 2))
    metric_col4.metric("Average Reward", round(summary["avg_reward"], 3))
    metric_col5.metric("Slate Reward@5", round(slate_reward, 4))

    if slate_quality is not None:
        st.markdown("### Multi-Agent Slate Quality@5")

        q_col1, q_col2, q_col3, q_col4, q_col5 = st.columns(5)

        q_col1.metric("Exact@5", slate_quality["exact_count_at_k"])
        q_col2.metric("Substitute@5", slate_quality["substitute_count_at_k"])
        q_col3.metric("Complement@5", slate_quality["complement_count_at_k"])
        q_col4.metric("Irrelevant@5", slate_quality["irrelevant_count_at_k"])
        q_col5.metric("Diversity@5", round(slate_quality["diversity_at_k"], 4))

    st.markdown("### Ranking Explanation")

    top_baseline = baseline_results.iloc[0]
    top_policy = feedback_results.iloc[0]

    st.info(f"{baseline_title} top result: {top_baseline['product_title']}")
    st.success(f"Final top result using '{selected_policy}': {top_policy['product_title']}")

    if contract_mode == "LLM Agent":
        st.markdown("### CORTEX Agent Explanation")
        st.write(contract.get("explanation", "No explanation provided."))


# =============================================================================
# Router Dashboard
# =============================================================================

def render_router_dashboard():
    st.markdown("## Router Dashboard")

    summary_df = load_csv_if_exists("outputs/router_integrated_scalable_eval_summary.csv")
    by_policy_df = load_csv_if_exists("outputs/router_integrated_scalable_eval_by_policy.csv")
    by_route_df = load_csv_if_exists("outputs/router_integrated_scalable_eval_by_route.csv")
    high_impact_df = load_csv_if_exists("outputs/router_integrated_scalable_eval_high_impact.csv")
    metadata = load_json_if_exists("models/learned_repair_router_metadata.json")

    if len(summary_df) == 0:
        st.warning(
            "Router-integrated scalable outputs not found yet. Run: "
            ".venv\\Scripts\\python.exe -m src.router_integrated_scalable_evaluator"
        )
        return

    row = summary_df.iloc[0]

    st.markdown(
        """
        <div class="info-card-blue">
            This dashboard visualizes the saved learned router as a dry-run policy layer.
            The router chooses among baseline, gated CORTEX, and full CORTEX per query.
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Rows Evaluated", int(row.get("query_count", 0)))
    c2.metric("Router Reward@5", round(safe_float(row.get("avg_router_reward_at_5")), 4))
    c3.metric("Gated Reward@5", round(safe_float(row.get("avg_gated_cortex_reward_at_5")), 4))
    c4.metric("Router vs Gated", f"+{safe_float(row.get('avg_router_delta_vs_gated')):.4f}")
    c5.metric("Near Oracle Rate", f"{safe_float(row.get('near_oracle_rate')) * 100:.1f}%")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        fig = reward_comparison_chart(summary_df)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)

    with chart_col2:
        fig = router_policy_distribution_chart(summary_df)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)

    fig = route_reward_chart(by_route_df)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)

    fig = router_3d_chart(by_policy_df)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Router Summary")
    st.dataframe(summary_df, use_container_width=True)

    st.markdown("### By Router Predicted Policy")
    st.dataframe(by_policy_df, use_container_width=True)

    st.markdown("### By Governance Route")
    st.dataframe(by_route_df, use_container_width=True)

    with st.expander("Saved Router Metadata", expanded=False):
        st.json(metadata)

    with st.expander("High Impact / Error Audit Rows", expanded=False):
        if len(high_impact_df) > 0:
            st.dataframe(high_impact_df.head(100), use_container_width=True)
        else:
            st.info("No high-impact rows found.")


# =============================================================================
# Learning Dashboard
# =============================================================================

def render_learning_dashboard():
    st.markdown("## Learning Dashboard")

    feedback_log = load_feedback_log()
    policy_summary = summarize_policy_rewards(feedback_log)
    q_table_df = get_q_table_as_dataframe()
    experiment_runs = load_experiment_runs()

    dash_col1, dash_col2 = st.columns(2)

    with dash_col1:
        st.markdown("### Bandit Policy Reward Summary")
        st.dataframe(policy_summary, use_container_width=True)

    with dash_col2:
        st.markdown("### Average Reward by Old Policy")

        if len(policy_summary) > 0 and "avg_reward" in policy_summary.columns:
            chart = px.bar(
                policy_summary,
                x="ranking_objective",
                y="avg_reward",
                text="avg_reward",
                title="Historical Average Reward per Policy",
                color="ranking_objective",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )

            chart.update_layout(
                template="plotly_white",
                showlegend=False,
            )

            st.plotly_chart(chart, use_container_width=True)
        else:
            st.info("No bandit policy reward data yet.")

    st.markdown("### Slate Q-Learning Q-Table")

    if len(q_table_df) > 0:
        st.dataframe(q_table_df, use_container_width=True)

        q_chart = px.bar(
            q_table_df,
            x="action",
            y="q_value",
            color="state",
            barmode="group",
            title="Contract-Conditioned Q-values by Action",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )

        q_chart.update_layout(template="plotly_white")

        st.plotly_chart(q_chart, use_container_width=True)
    else:
        st.info("No Q-learning updates yet. Run Slate Q-Learning mode and save feedback.")

    st.markdown("### Experiment Snapshot Log")

    if len(experiment_runs) > 0:
        display_cols = [
            "timestamp",
            "query",
            "retrieval_mode",
            "contract_mode",
            "policy_mode",
            "selected_policy",
            "contract_source",
            "product_type",
            "preferred_brand",
            "enforcement_status",
            "low_coverage",
            "top_product_title",
            "slate_reward_at_5",
        ]

        st.dataframe(
            compact_cols(experiment_runs.tail(50), display_cols),
            use_container_width=True,
        )

        if "slate_reward_at_5" in experiment_runs.columns:
            exp_chart = px.line(
                experiment_runs.reset_index(),
                x="index",
                y="slate_reward_at_5",
                title="SlateReward@5 Across Saved Experiment Runs",
                markers=True,
            )

            exp_chart.update_layout(template="plotly_white")

            st.plotly_chart(exp_chart, use_container_width=True)
    else:
        st.info("No experiment snapshots saved yet.")

    st.markdown("### Recent Feedback Log")

    if len(feedback_log) > 0:
        st.dataframe(feedback_log.tail(30), use_container_width=True)

        metric_col1, metric_col2, metric_col3 = st.columns(3)

        metric_col1.metric("Logged Rows", len(feedback_log))
        metric_col2.metric("Total Logged Clicks", int(feedback_log["simulated_click"].sum()))
        metric_col3.metric(
            "Total Logged Reward",
            round(float(feedback_log["simulated_reward"].sum()), 2),
        )
    else:
        st.info("No feedback has been saved yet.")


# =============================================================================
# Architecture tab
# =============================================================================

def render_architecture_page():
    st.markdown("## Architecture")

    st.markdown(
        """
        <div class="info-card-blue">
            CORTEX is not just a reranker. It is a governed agentic ranking system
            that uses different decision layers to decide how aggressive or conservative
            ranking should be for a given query.
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_agent_flow()

    st.markdown("### Current MVP Milestones")

    milestones = pd.DataFrame(
        [
            ["MVP 13.3", "Baseline Preservation Gate", "Prevents unnecessary over-reranking."],
            ["MVP 13.5", "Agent Governance Controller", "Routes queries by risk, cost, and agent need."],
            ["MVP 13.6", "Critic / Verifier Agent", "Diagnoses failure modes and repair actions."],
            ["MVP 13.7", "Repair Simulator", "Estimates repair value from logged outcomes."],
            ["MVP 13.8", "Learned Repair Router", "Predicts baseline vs gated vs full CORTEX."],
            ["MVP 13.9", "Out-of-Sample Validation", "Tests router generalization on held-out queries."],
            ["MVP 14.1", "Saved Router Model", "Persists model, feature schema, and metadata."],
            ["MVP 14.2", "Inference Smoke Test", "Loads saved model and scores rows."],
            ["MVP 14.3", "Router-Integrated Scalable Evaluator", "Formal dry-run strategy comparison."],
        ],
        columns=["MVP", "Component", "Purpose"],
    )

    st.dataframe(milestones, use_container_width=True)

    st.markdown("### Roadmap")

    roadmap = pd.DataFrame(
        [
            ["MVP 14.4", "Streamlit Router Dry-Run Toggle", "Add router policy option to the live demo UI."],
            ["MVP 15", "Mission-Based Shopping Agent", "Decompose intent like 'World Cup watch party' into item bundles."],
            ["MVP 16", "Behavior-Aware CORTEX", "Use clicks, purchases, ATC, and reward feedback."],
            ["MVP 17", "Multimodal CORTEX", "Use image/text/product metadata for richer ranking decisions."],
            ["MVP 18", "Online Learning Loop", "Update routing and ranking policies from new feedback."],
        ],
        columns=["Stage", "Planned Capability", "Why It Matters"],
    )

    st.dataframe(roadmap, use_container_width=True)


# =============================================================================
# Main app
# =============================================================================

products = load_products()

with st.sidebar:
    st.markdown("## CORTEX Controls")

    st.markdown(
        """
        <div class="small-muted">
            Recommended live setting:
            <br><b>Semantic + LLM Agent + Slate Q-Learning + Multi-Agent Diversification</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("### Project State")
    st.write("MVP 14.3 completed")
    st.write("Router dry-run evaluator ready")
    st.write("Saved router model ready")

    st.divider()

    st.markdown("### Useful Commands")
    st.code(
        ".venv\\Scripts\\streamlit.exe run app.py\n"
        ".venv\\Scripts\\python.exe -m src.router_integrated_scalable_evaluator\n"
        "git status",
        language="powershell",
    )

render_header()

tab_search, tab_router, tab_learning, tab_architecture = st.tabs(
    [
        "Live Search Console",
        "Router Dashboard",
        "Learning Dashboard",
        "Architecture",
    ]
)

with tab_search:
    render_search_console(products)

with tab_router:
    render_router_dashboard()

with tab_learning:
    render_learning_dashboard()

with tab_architecture:
    render_architecture_page()