import os
import json
import pickle
import subprocess
import sys
from datetime import datetime
from html import escape

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


ROUTER_MODEL_PATH = "models/learned_repair_router.pkl"
ROUTER_FEATURES_PATH = "models/learned_repair_router_features.json"
ROUTER_METADATA_PATH = "models/learned_repair_router_metadata.json"
ROUTER_DRY_RUN_LOG_PATH = "storage/router_dry_run_log.csv"
GOVERNED_SUMMARY_PATH = "outputs/governed_cortex_summary.csv"
GOVERNED_FINAL_SLATE_PATH = "outputs/governed_cortex_final_slate.csv"
GOVERNED_TRACE_PATH = "outputs/governed_cortex_trace.csv"
GOVERNANCE_DECISIONS_PATH = "outputs/cortex_governance_decisions.csv"
GOVERNANCE_SUMMARY_PATH = "outputs/cortex_governance_summary.csv"
SCALABLE_GOVERNED_EVAL_SUMMARY_PATH = "outputs/scalable_governed_eval_summary.csv"
SCALABLE_GOVERNED_EVAL_PATH = "outputs/scalable_governed_eval.csv"

ROUTE_LABELS = {
    "BASELINE_ONLY": "Preserve baseline",
    "MISSION_BUILD": "Build mission slate",
    "MISSION_REPAIR": "Repair mission slate",
    "STRICT_REPAIR": "Strict repair with guardrails",
    "BEHAVIOR_AWARE_RERANK": "Behavior-aware CORTEX",
    "CRITIC_REVIEW": "Send to critic review",
    "REJECT_REPAIR_NARROW_QUERY": "Block repair for narrow query",
}

EXECUTION_LABELS = {
    "behavior_aware": "Behavior-aware CORTEX executed",
    "strict_repair": "Strict repaired slate executed",
    "baseline_fallback": "Baseline-style fallback preserved",
    "fallback_no_slate_available": "Safe fallback used",
}


# =============================================================================
# Polished CORTEX UI styling
# =============================================================================

st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(59, 130, 246, 0.12), transparent 26%),
                radial-gradient(circle at top right, rgba(16, 185, 129, 0.10), transparent 28%),
                linear-gradient(180deg, #f7fbff 0%, #f8fafc 45%, #ffffff 100%);
            color: #0f172a;
        }

        section[data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid #e2e8f0;
        }

        [data-testid="stSidebarNav"] {
            display: none;
        }

        .main-title-card {
            background: linear-gradient(135deg, #ffffff 0%, #eef6ff 48%, #ecfdf5 100%);
            border: 1px solid #dbeafe;
            border-radius: 26px;
            padding: 30px 34px;
            margin-bottom: 18px;
            box-shadow: 0 18px 45px rgba(15, 23, 42, 0.08);
        }

        .main-eyebrow {
            font-size: 0.82rem;
            font-weight: 850;
            color: #2563eb;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            margin-bottom: 8px;
        }

        .main-title {
            font-size: 2.55rem;
            font-weight: 900;
            color: #0f172a;
            letter-spacing: -0.055em;
            line-height: 1.08;
            margin-bottom: 8px;
        }

        .main-subtitle {
            font-size: 1.02rem;
            color: #475569;
            line-height: 1.65;
            max-width: 1120px;
        }

        .badge-row {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            margin-top: 16px;
        }

        .badge {
            font-size: 0.78rem;
            font-weight: 800;
            padding: 8px 12px;
            border-radius: 999px;
            color: #1e293b;
            background: #ffffff;
            border: 1px solid #cbd5e1;
            box-shadow: 0 4px 10px rgba(15, 23, 42, 0.04);
        }

        .soft-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 20px;
            padding: 18px 20px;
            margin-bottom: 16px;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.055);
        }

        .demo-hero {
            text-align: center;
            max-width: 880px;
            margin: 22px auto 22px auto;
        }

        .demo-hero-title {
            color: #0f172a;
            font-size: 2.35rem;
            line-height: 1.1;
            font-weight: 900;
            letter-spacing: -0.055em;
            margin-bottom: 10px;
        }

        .demo-hero-subtitle {
            color: #475569;
            line-height: 1.65;
            font-size: 1.01rem;
        }

        .search-card {
            max-width: 820px;
            margin: 0 auto 22px auto;
            padding: 22px 26px 12px 26px;
            background: #ffffff;
            border: 1px solid #dbeafe;
            border-radius: 24px;
            box-shadow: 0 14px 36px rgba(15, 23, 42, 0.07);
        }

        .decision-card {
            background: linear-gradient(135deg, #ffffff 0%, #eff6ff 100%);
            border: 1px solid #bfdbfe;
            border-radius: 24px;
            padding: 25px 28px;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.065);
            margin: 18px 0;
        }

        .decision-label {
            color: #2563eb;
            font-size: 0.76rem;
            font-weight: 850;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-bottom: 7px;
        }

        .decision-title {
            color: #0f172a;
            font-size: 1.9rem;
            line-height: 1.15;
            font-weight: 900;
            letter-spacing: -0.04em;
            margin-bottom: 10px;
        }

        .simple-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 20px;
            padding: 19px 20px;
            min-height: 178px;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
        }

        .simple-card-title {
            color: #0f172a;
            font-weight: 850;
            font-size: 1rem;
            margin-bottom: 9px;
        }

        .simple-card-body {
            color: #475569;
            line-height: 1.62;
            font-size: 0.94rem;
        }

        .metric-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 18px;
            min-height: 108px;
            padding: 14px 16px;
            margin-bottom: 16px;
            box-shadow: 0 7px 18px rgba(15, 23, 42, 0.05);
        }

        .metric-card-label {
            color: #64748b;
            font-size: 0.72rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .metric-card-value {
            color: #0f172a;
            font-size: 1.45rem;
            font-weight: 900;
            margin: 7px 0 4px;
        }

        .metric-card-help {
            color: #64748b;
            font-size: 0.82rem;
        }

        .product-card {
            background: #ffffff;
            border: 1px solid #dbeafe;
            border-left: 5px solid #2563eb;
            border-radius: 17px;
            padding: 14px 17px;
            margin: 10px 0;
            box-shadow: 0 6px 16px rgba(15, 23, 42, 0.045);
        }

        .product-rank {
            color: #2563eb;
            font-size: 0.73rem;
            font-weight: 850;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .product-title {
            color: #0f172a;
            font-weight: 850;
            font-size: 1.02rem;
            margin: 5px 0 7px;
        }

        .product-meta {
            color: #475569;
            font-size: 0.88rem;
            line-height: 1.55;
        }

        .info-card-blue {
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            border-radius: 18px;
            padding: 16px 18px;
            color: #1e3a8a;
            margin-bottom: 12px;
            line-height: 1.62;
        }

        .info-card-green {
            background: #ecfdf5;
            border: 1px solid #bbf7d0;
            border-radius: 18px;
            padding: 16px 18px;
            color: #065f46;
            margin-bottom: 12px;
            line-height: 1.62;
        }

        .info-card-yellow {
            background: #fffbeb;
            border: 1px solid #fde68a;
            border-radius: 18px;
            padding: 16px 18px;
            color: #92400e;
            margin-bottom: 12px;
            line-height: 1.62;
        }

        .info-card-purple {
            background: #f5f3ff;
            border: 1px solid #ddd6fe;
            border-radius: 18px;
            padding: 16px 18px;
            color: #4c1d95;
            margin-bottom: 12px;
            line-height: 1.62;
        }

        .agent-box {
            background: #ffffff;
            border: 1px solid #dbeafe;
            border-radius: 16px;
            padding: 14px;
            text-align: center;
            font-weight: 800;
            color: #1e3a8a;
            min-height: 72px;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 6px 14px rgba(37, 99, 235, 0.06);
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
            font-weight: 700;
        }

        div[data-testid="stMetricValue"] {
            color: #0f172a;
            font-weight: 900;
        }

        .small-muted {
            color: #64748b;
            font-size: 0.92rem;
            line-height: 1.6;
        }

        .section-kicker {
            color: #2563eb;
            font-size: 0.78rem;
            font-weight: 850;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            margin-bottom: 6px;
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

        h1, h2, h3 {
            color: #0f172a;
            letter-spacing: -0.025em;
        }

        .block-container {
            padding-top: 2.3rem;
        }

        div[data-testid="stDataFrame"] {
            border-radius: 16px;
            overflow: hidden;
        }

        /* MVP 19.2C dark product shell */
        .stApp {
            background:
                radial-gradient(circle at 16% 0%, rgba(6, 182, 212, 0.16), transparent 27%),
                radial-gradient(circle at 90% 7%, rgba(124, 58, 237, 0.15), transparent 29%),
                linear-gradient(145deg, #030712 0%, #071326 45%, #020617 100%);
            color: #e5edf8;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #07101d 0%, #050b16 100%);
            border-right: 1px solid #192b42;
        }

        section[data-testid="stSidebar"] * {
            color: #d9e6f5;
        }

        .main-title-card, .decision-card {
            background: linear-gradient(135deg, rgba(11, 25, 45, 0.98), rgba(12, 35, 59, 0.96));
            border: 1px solid rgba(34, 211, 238, 0.20);
            box-shadow: 0 18px 48px rgba(0, 0, 0, 0.34), 0 0 40px rgba(6, 182, 212, 0.06);
        }

        .main-eyebrow, .decision-label, .section-kicker, .product-rank {
            color: #22d3ee;
        }

        .main-title, .decision-title, .demo-hero-title, .simple-card-title,
        .product-title, h1, h2, h3 {
            color: #f8fafc;
        }

        .main-subtitle, .demo-hero-subtitle, .small-muted, .simple-card-body,
        .product-meta, .metric-card-help {
            color: #a6b8cf;
        }

        .badge {
            color: #bae6fd;
            background: rgba(14, 36, 60, 0.74);
            border-color: rgba(34, 211, 238, 0.20);
        }

        .soft-card, .simple-card, .metric-card, div[data-testid="stMetric"] {
            background: rgba(10, 22, 39, 0.94);
            border: 1px solid #1b314a;
            box-shadow: 0 10px 26px rgba(0, 0, 0, 0.23);
        }

        .metric-card-label, div[data-testid="stMetricLabel"] {
            color: #8ca5c0;
        }

        .metric-card-value, div[data-testid="stMetricValue"] {
            color: #f1f5f9;
        }

        .product-card {
            background: rgba(9, 22, 39, 0.96);
            border: 1px solid #1c3850;
            border-left: 5px solid #22d3ee;
            box-shadow: 0 9px 23px rgba(0, 0, 0, 0.24);
        }

        .info-card-blue, .info-card-green, .info-card-yellow, .info-card-purple {
            background: rgba(9, 23, 42, 0.92);
            color: #d5e3f4;
        }

        .info-card-blue { border-color: rgba(34, 211, 238, 0.30); }
        .info-card-green { border-color: rgba(16, 185, 129, 0.34); }
        .info-card-yellow { border-color: rgba(245, 158, 11, 0.36); }
        .info-card-purple { border-color: rgba(139, 92, 246, 0.36); }

        .agent-box {
            background: #0b182b;
            border-color: #1e3a55;
            color: #d8f5ff;
        }

        .agent-arrow {
            color: #22d3ee;
        }

        .stTabs [data-baseweb="tab"] {
            background: rgba(9, 20, 36, 0.94);
            border-color: #1c3148;
            color: #afc2d8;
        }

        .stTabs [aria-selected="true"] {
            background: rgba(8, 65, 87, 0.78);
            border-color: #22d3ee;
            color: #ecfeff;
            box-shadow: 0 0 18px rgba(34, 211, 238, 0.18);
        }

        .search-hero {
            text-align: center;
            max-width: 900px;
            margin: 20px auto 14px;
        }

        .search-hero-title {
            color: #f8fafc;
            font-size: clamp(2.2rem, 4vw, 3.3rem);
            font-weight: 900;
            letter-spacing: -0.06em;
            line-height: 1.05;
            margin: 8px 0 12px;
        }

        .search-hero-subtitle {
            color: #b0c2d7;
            font-size: 1.02rem;
            line-height: 1.65;
        }

        .chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 9px;
            justify-content: center;
            margin: 4px auto 26px;
        }

        .query-chip {
            color: #b8d8ea;
            background: rgba(10, 29, 49, 0.80);
            border: 1px solid #20405b;
            padding: 7px 13px;
            border-radius: 999px;
            font-size: 0.83rem;
        }

        .stTextInput input {
            color: #f8fafc !important;
            background: #0a172b !important;
            border: 1px solid #284b68 !important;
            border-radius: 14px !important;
            min-height: 52px;
        }

        .stTextInput input:focus {
            border-color: #22d3ee !important;
            box-shadow: 0 0 0 1px #22d3ee, 0 0 22px rgba(34, 211, 238, 0.18) !important;
        }

        .stButton > button[kind="primary"] {
            background: linear-gradient(90deg, #0891b2, #2563eb);
            border: 0;
            color: #ffffff;
            font-weight: 750;
            min-height: 46px;
            border-radius: 12px;
            box-shadow: 0 8px 22px rgba(37, 99, 235, 0.28);
        }

        /* Streamlit widget contrast on the dark canvas. */
        .stApp label,
        .stApp p,
        .stApp span,
        .stApp div[data-testid="stMarkdownContainer"],
        .stApp div[data-testid="stMarkdownContainer"] p,
        .stApp div[data-testid="stWidgetLabel"],
        .stApp div[data-testid="stWidgetLabel"] p,
        .stApp .stRadio label,
        .stApp .stCheckbox label,
        .stApp .stSelectbox label,
        .stApp .stTextInput label,
        .stApp div[data-testid="stRadio"] label,
        .stApp div[data-testid="stRadio"] label p,
        .stApp div[data-testid="stCheckbox"] label,
        .stApp div[data-testid="stCheckbox"] label p,
        .stApp div[data-testid="stSelectbox"] label,
        .stApp div[data-testid="stTextInput"] label,
        .stApp div[data-testid="stSlider"] label {
            color: #cbd5e1 !important;
        }

        .stApp h1, .stApp h2, .stApp h3, .stApp h4,
        .stApp h5, .stApp h6,
        .stApp div[data-testid="stMarkdownContainer"] strong {
            color: #f8fafc;
        }

        .stApp div[data-testid="stRadio"] label p,
        .stApp .stRadio label p,
        .stApp div[data-testid="stCheckbox"] label p,
        .stApp .stCheckbox label p {
            color: #e5edf8 !important;
        }

        .stApp div[data-testid="stTextInput"] input,
        .stApp .stTextInput input {
            color: #f8fafc !important;
            caret-color: #38bdf8;
            background-color: #0f172a !important;
            border-color: #334155 !important;
        }

        .stApp div[data-testid="stTextInput"] input::placeholder,
        .stApp .stTextInput input::placeholder {
            color: #94a3b8 !important;
            opacity: 1;
        }

        .stApp div[data-baseweb="select"],
        .stApp div[data-baseweb="select"] > div,
        .stApp div[data-baseweb="select"] * {
            color: #f8fafc !important;
        }

        .stApp div[data-baseweb="select"] > div {
            background-color: #0f172a !important;
            border-color: #334155 !important;
        }

        .stApp div[data-baseweb="popover"],
        .stApp div[data-baseweb="menu"],
        .stApp ul[role="listbox"] {
            background-color: #111827 !important;
            color: #f8fafc !important;
        }

        .stApp li[role="option"],
        .stApp li[role="option"] * {
            color: #f8fafc !important;
        }

        .stApp li[role="option"]:hover,
        .stApp li[role="option"][aria-selected="true"] {
            background-color: #1e3a5f !important;
        }

        .stApp div[data-testid="stExpander"] details,
        .stApp div[data-testid="stExpander"] summary {
            background: rgba(10, 22, 39, 0.76);
            color: #e5edf8 !important;
        }

        .stApp div[data-testid="stExpander"] summary *,
        .stApp div[data-testid="stExpander"] details > div * {
            color: #e5edf8;
        }

        .stApp .stTabs [data-baseweb="tab"] *,
        .stApp .stTabs [data-baseweb="tab"] p {
            color: inherit !important;
        }

        .stApp div[data-testid="stSpinner"] *,
        .stApp div[data-testid="stStatusWidget"] *,
        .stApp div[data-testid="stAlert"] p {
            color: #e5edf8 !important;
        }

        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] div[data-testid="stWidgetLabel"] {
            color: #cbd5e1 !important;
        }

        /* Keep embedded grid cells free to use Streamlit's high-contrast dataframe theme. */
        .stApp div[data-testid="stDataFrame"] span,
        .stApp div[data-testid="stDataFrame"] p {
            color: inherit !important;
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


@st.cache_resource(show_spinner=False)
def load_router_model():
    if not os.path.exists(ROUTER_MODEL_PATH):
        return None

    with open(ROUTER_MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner=False)
def load_router_feature_payload():
    return load_json_if_exists(ROUTER_FEATURES_PATH)


@st.cache_data(show_spinner=False)
def load_router_metadata():
    return load_json_if_exists(ROUTER_METADATA_PATH)


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def safe_numeric_series(series):
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def compact_cols(df, cols):
    existing = [col for col in cols if col in df.columns]
    if not existing:
        return df
    return df[existing]


def get_top_product_title(df):
    if df is None or len(df) == 0:
        return ""

    for col in ["product_title", "title"]:
        if col in df.columns:
            return str(df.iloc[0].get(col, ""))

    return ""


def render_header():
    st.markdown(
        """
        <div class="main-title-card">
            <div class="main-eyebrow">PolicyRank-RL · CORTEX Engine</div>
            <div class="main-title">Agentic search ranking with governed policy intelligence.</div>
            <div class="main-subtitle">
                A clean research prototype for semantic retrieval, LLM-generated search contracts,
                contract-aware filtering, slate Q-learning, multi-agent diversification, critic verification,
                learned repair routing, mission-aware shopping, and behavior-aware cold-start rescue.
            </div>
            <div class="badge-row">
                <div class="badge">Semantic Retrieval</div>
                <div class="badge">LLM Contracts</div>
                <div class="badge">Slate Q-Learning</div>
                <div class="badge">Mission Repair</div>
                <div class="badge">Behavior-Aware CORTEX</div>
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
# Product Home / Renovated Dashboard Helpers
# =============================================================================

def render_capability_card(title, subtitle, accent="#2563eb"):
    st.markdown(
        f"""
        <div class="soft-card" style="border-left: 7px solid {accent}; min-height: 145px;">
            <div style="font-size: 1.02rem; font-weight: 850; color: #0f172a; margin-bottom: 8px;">
                {title}
            </div>
            <div class="small-muted">
                {subtitle}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_pill(label, value, accent="#2563eb"):
    st.markdown(
        f"""
        <div class="soft-card" style="border-left: 7px solid {accent};">
            <div style="font-size: 0.78rem; font-weight: 800; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em;">
                {label}
            </div>
            <div style="font-size: 1.35rem; font-weight: 900; color: #0f172a; margin-top: 8px;">
                {value}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_product_home():
    st.markdown("## CORTEX Product Home")

    st.markdown(
        """
        <div class="info-card-blue">
            <b>CORTEX</b> is an agentic ranking prototype for search. It does not only retrieve and rank products.
            It reasons about user intent, contracts, slate quality, mission coverage, critic feedback,
            repair decisions, behavior confidence, cold-start rescue, and routing policy.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Current System Snapshot")

    s1, s2, s3, s4 = st.columns(4)

    with s1:
        render_status_pill("Current Stage", "MVP 16.2", "#2563eb")
    with s2:
        render_status_pill("Core Mode", "Agentic Ranking", "#7c3aed")
    with s3:
        render_status_pill("Latest Layer", "Behavior-Aware", "#059669")
    with s4:
        render_status_pill("Repo Flow", "Build → Test → Commit", "#d97706")

    st.markdown("### What CORTEX Does")

    c1, c2, c3 = st.columns(3)

    with c1:
        render_capability_card(
            "Intent and Contract Intelligence",
            "Turns a query into a search contract using rule-based or LLM-assisted logic, then filters candidates against the contract.",
            "#2563eb",
        )

    with c2:
        render_capability_card(
            "Policy Ranking and Slate Governance",
            "Uses policy ranking, slate enforcement, Q-learning actions, and diversification to produce a controlled final slate.",
            "#7c3aed",
        )

    with c3:
        render_capability_card(
            "Behavior-Aware Mission Repair",
            "Balances behavior evidence with mission-stage coverage, cold-start rescue, exploration, and explainable reason codes.",
            "#059669",
        )

    st.markdown("### Agentic Flow")

    render_agent_flow()

    st.markdown("### MVP Progress Timeline")

    milestones = pd.DataFrame(
        [
            ["MVP 13.x", "Governed CORTEX Foundation", "Baseline gate, critic, repair simulator, learned router."],
            ["MVP 14.x", "Router Stack", "Saved router model, scalable evaluator, Streamlit dry-run, decision logging, analyzer."],
            ["MVP 15.1", "Mission Agent", "Detects mission-like shopping queries."],
            ["MVP 15.2", "Mission Slate Builder", "Builds multi-intent shopping slates."],
            ["MVP 15.3", "Mission Relevance Guardrails", "Protects against weak or irrelevant mission results."],
            ["MVP 15.4", "Streamlit Mission Integration", "Adds mission-shopping integration to the UI."],
            ["MVP 15.5", "Mission Coverage Analyzer", "Measures sub-intent coverage and gaps."],
            ["MVP 15.6", "Mission Critic Agent", "Diagnoses slate weakness and repair needs."],
            ["MVP 15.7", "Mission Repair Loop", "Adds missing mission sub-intents when useful."],
            ["MVP 15.8", "Repair Quality Guardrails", "Rejects weak repaired candidates."],
            ["MVP 15.9", "Strict Compound Repair Rules", "Prevents over-repair for narrow queries and validates compound intent repairs."],
            ["MVP 16", "Behavior-Aware CORTEX", "Adds behavior confidence, mission stage, exploration, cold-start rescue, and policy reasons."],
            ["MVP 16.1", "Behavior-Aware Streamlit Page", "Makes the behavior-aware layer interactive."],
            ["MVP 16.2", "UI Renovation", "Turns the app into a cleaner, demo-ready CORTEX product dashboard."],
        ],
        columns=["Stage", "Capability", "What It Adds"],
    )

    st.dataframe(milestones, use_container_width=True, hide_index=True)

    st.markdown("### Demo Playbook")

    d1, d2 = st.columns(2)

    with d1:
        st.markdown(
            """
            <div class="info-card-green">
                <b>Best demo path:</b><br><br>
                1. Start with this Product Home tab.<br>
                2. Open the Behavior-Aware CORTEX page from the sidebar.<br>
                3. Run <b>beach vacation packing list</b>.<br>
                4. Show cold-start rescue and policy reason codes.<br>
                5. Return to Live Search Console for the broader CORTEX pipeline.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with d2:
        st.markdown(
            """
            <div class="info-card-purple">
                <b>Best query examples:</b><br><br>
                <b>new apartment kitchen setup</b> → compound setup mission<br>
                <b>beach vacation packing list</b> → multi-category travel mission<br>
                <b>adidas soccer cleats</b> → narrow branded product query<br>
                <b>world cup watch party</b> → event-based shopping mission
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Why This Is Different")

    st.markdown(
        """
        <div class="info-card-yellow">
            A normal ranking model can become click-greedy: it may over-promote items with strong behavior evidence
            and bury useful but underexposed items. CORTEX adds a policy layer that asks:
            <b>Does this item help complete the user's mission?</b>
            That is why the latest behavior-aware layer includes cold-start rescue, exploration flags,
            and policy reason codes.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# Governed CORTEX demo helpers
# =============================================================================

def run_governed_cortex_demo(query, fast_mode):
    cmd = [
        sys.executable,
        "-m",
        "src.governed_cortex_runner",
        "--query",
        query,
    ]

    if fast_mode:
        cmd.append("--skip-refresh")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except OSError as exc:
        return False, str(exc)

    output = result.stdout or ""
    if result.stderr:
        output += f"\n\nSTDERR:\n{result.stderr}"

    load_csv_if_exists.clear()
    return result.returncode == 0, output


def filter_by_query(df, query):
    if df is None or df.empty or not query or "query" not in df.columns:
        return pd.DataFrame() if df is None else df

    query_lower = str(query).strip().lower()
    return df[df["query"].astype(str).str.strip().str.lower() == query_lower].copy()


def get_latest_row(df):
    if df is None or df.empty:
        return {}
    return df.iloc[-1].to_dict()


def friendly_route(route):
    return ROUTE_LABELS.get(str(route), str(route).replace("_", " ").title())


def friendly_execution(source):
    return EXECUTION_LABELS.get(str(source), str(source).replace("_", " ").title())


def _demo_int(value):
    try:
        return int(float(value))
    except Exception:
        return 0


def build_simple_decision(summary_row):
    if not summary_row:
        return "Run CORTEX to generate a governed decision."
    if _demo_int(summary_row.get("baseline_preserved")) == 1:
        return "CORTEX preserved baseline-style handling."
    if summary_row.get("final_execution_source") == "behavior_aware":
        return "CORTEX intervened with behavior-aware mission ranking."
    if summary_row.get("final_execution_source") == "strict_repair":
        return "CORTEX intervened with strict repair guardrails."
    return f"CORTEX selected {friendly_route(summary_row.get('governance_route', 'unknown'))}."


def build_simple_why(summary_row):
    if not summary_row:
        return "The governance explanation appears after a query is run."

    if _demo_int(summary_row.get("baseline_preserved")) == 1:
        return (
            "The governance layer decided aggressive repair was unnecessary or risky for this "
            "query, so it protected the baseline path."
        )

    mission = safe_float(summary_row.get("mission_likelihood_score"))
    risk = safe_float(summary_row.get("repair_risk_score"))
    coverage = safe_float(summary_row.get("coverage_gap_score"))
    return (
        f"Mission likelihood was {mission:.2f}, repair risk was {risk:.2f}, and coverage gap "
        f"was {coverage:.2f}. CORTEX used those signals to select a governed route."
    )


def build_business_meaning(summary_row):
    if not summary_row:
        return "Business impact appears after running CORTEX."

    if _demo_int(summary_row.get("baseline_preserved")) == 1:
        return (
            "CORTEX avoided unnecessary compute and ranking risk by declining an aggressive "
            "intervention where it was not justified."
        )

    slate_size = _demo_int(summary_row.get("final_slate_size"))
    needs = _demo_int(summary_row.get("unique_sub_intents"))
    rescue = _demo_int(summary_row.get("cold_start_proxy_items"))
    return (
        f"The result covers {needs} shopping needs across {slate_size} governed items and "
        f"retains {rescue} potentially useful low-evidence items."
    )


def prepare_simple_slate(slate_df):
    if slate_df is None or slate_df.empty:
        return pd.DataFrame()

    cols = [
        "governed_rank",
        "product_title",
        "sub_intent",
        "mission_stage",
        "behavior_confidence",
        "policy_reason",
        "final_policy_score",
    ]
    out = compact_cols(slate_df, cols).copy()
    out = out.rename(
        columns={
            "governed_rank": "Rank",
            "product_title": "Product",
            "sub_intent": "Need / Sub-Intent",
            "mission_stage": "Mission Stage",
            "behavior_confidence": "Confidence",
            "policy_reason": "Reason",
            "final_policy_score": "Score",
        }
    )
    if "Score" in out.columns:
        out["Score"] = pd.to_numeric(out["Score"], errors="coerce").round(4)
    return out


def _render_demo_metric(label, value, help_text):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-card-label">{escape(str(label))}</div>
            <div class="metric-card-value">{escape(str(value))}</div>
            <div class="metric-card-help">{escape(str(help_text))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_demo_product_cards(slate_df):
    if slate_df is None or slate_df.empty:
        st.info("No final governed slate rows are available for this query.")
        return

    for row in slate_df.head(6).to_dict(orient="records"):
        st.markdown(
            f"""
            <div class="product-card">
                <div class="product-rank">Rank {escape(str(_demo_int(row.get("governed_rank"))))}</div>
                <div class="product-title">{escape(str(row.get("product_title", "Untitled result")))}</div>
                <div class="product-meta">
                    <b>Need:</b> {escape(str(row.get("sub_intent", "")))}<br>
                    <b>Reason:</b> {escape(str(row.get("policy_reason", "")))}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_cortex_demo():
    st.markdown(
        """
        <div class="demo-hero">
            <div class="section-kicker">Governed Search Experience</div>
            <div class="demo-hero-title">What are you shopping for today?</div>
            <div class="demo-hero-subtitle">
                Enter any search query. CORTEX decides whether to preserve a stable baseline
                or execute governed mission-aware ranking, then explains the result.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    query_col = st.columns([1, 2.2, 1])[1]
    with query_col:
        with st.container(border=True):
            st.markdown(
                '<div class="section-kicker" style="text-align:center;">Try CORTEX</div>',
                unsafe_allow_html=True,
            )
            query = st.text_input(
                "Shopping query",
                placeholder="Example: beach vacation packing list",
                key="cortex_demo_query",
                label_visibility="collapsed",
            ).strip()
            button_col = st.columns([1, 1.05, 1])[1]
            with button_col:
                run_clicked = st.button(
                    "Run CORTEX",
                    type="primary",
                    use_container_width=True,
                    key="run_cortex_demo",
                )

    if run_clicked:
        if not query:
            st.warning("Enter a query before running CORTEX.")
        else:
            with st.spinner(f"Running governed CORTEX for '{query}'..."):
                ok, output = run_governed_cortex_demo(
                    query=query,
                    fast_mode=st.session_state.get("cortex_fast_mode", False),
                )
            st.session_state["cortex_demo_raw_output"] = output
            st.session_state["cortex_demo_last_query"] = query
            if ok:
                st.success("CORTEX completed. Governed outputs have been refreshed.")
            else:
                st.error("CORTEX could not complete. Switch to Technical Mode to inspect backend output.")

    active_query = query or st.session_state.get("cortex_demo_last_query", "")
    if not active_query:
        st.markdown(
            '<div class="info-card-blue">Enter any query and run CORTEX to see a clean governed decision.</div>',
            unsafe_allow_html=True,
        )
        return

    summary_df = filter_by_query(load_csv_if_exists(GOVERNED_SUMMARY_PATH), active_query)
    slate_df = filter_by_query(load_csv_if_exists(GOVERNED_FINAL_SLATE_PATH), active_query)
    trace_df = filter_by_query(load_csv_if_exists(GOVERNED_TRACE_PATH), active_query)
    decisions_df = filter_by_query(load_csv_if_exists(GOVERNANCE_DECISIONS_PATH), active_query)
    governance_summary_df = filter_by_query(load_csv_if_exists(GOVERNANCE_SUMMARY_PATH), active_query)
    scalable_summary_df = load_csv_if_exists(SCALABLE_GOVERNED_EVAL_SUMMARY_PATH)
    scalable_eval_df = load_csv_if_exists(SCALABLE_GOVERNED_EVAL_PATH)

    if "governed_rank" in slate_df.columns:
        slate_df["governed_rank"] = pd.to_numeric(slate_df["governed_rank"], errors="coerce")
        slate_df = slate_df.sort_values("governed_rank")

    summary_row = get_latest_row(summary_df)
    if not summary_row:
        st.info("No governed result exists for this query yet. Select Run CORTEX to generate it.")
        return

    if st.session_state.get("cortex_demo_mode", "Simple Mode") == "Simple Mode":
        route = friendly_route(summary_row.get("governance_route", "unknown"))
        execution = friendly_execution(summary_row.get("final_execution_source", "unknown"))
        st.markdown(
            f"""
            <div class="decision-card">
                <div class="decision-label">CORTEX Decision</div>
                <div class="decision-title">{escape(build_simple_decision(summary_row))}</div>
                <div class="small-muted"><b>Route:</b> {escape(route)}<br>
                <b>Execution source:</b> {escape(execution)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            _render_demo_metric("Final Slate", _demo_int(summary_row.get("final_slate_size")), "Governed results")
        with m2:
            _render_demo_metric("Needs Covered", _demo_int(summary_row.get("unique_sub_intents")), "Distinct sub-intents")
        with m3:
            _render_demo_metric("Baseline Preserved", _demo_int(summary_row.get("baseline_preserved")), "1 means intervention blocked")
        with m4:
            _render_demo_metric("Cold-Start Rescue", _demo_int(summary_row.get("cold_start_proxy_items")), "Useful low-evidence items")

        why_col, business_col = st.columns(2)
        with why_col:
            st.markdown(
                f'<div class="simple-card"><div class="simple-card-title">Why CORTEX chose this route</div>'
                f'<div class="simple-card-body">{escape(build_simple_why(summary_row))}</div></div>',
                unsafe_allow_html=True,
            )
        with business_col:
            st.markdown(
                f'<div class="simple-card"><div class="simple-card-title">Business meaning</div>'
                f'<div class="simple-card-body">{escape(build_business_meaning(summary_row))}</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("### Final Governed Slate")
        render_demo_product_cards(slate_df)
        with st.expander("View as table", expanded=False):
            st.dataframe(prepare_simple_slate(slate_df), use_container_width=True, hide_index=True)
        return

    st.markdown("### Technical Mode")
    tech_summary, tech_signals, tech_slate, tech_eval, tech_raw = st.tabs(
        ["Governed Summary", "Governance Signals", "Final Slate", "Scalable Evaluation", "Raw Trace"]
    )
    with tech_summary:
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    with tech_signals:
        signal_cols = [
            "query_type", "recommended_route", "governance_decision",
            "mission_likelihood_score", "compound_intent_score", "brand_specificity_score",
            "narrow_query_score", "repair_risk_score", "coverage_gap_score",
            "behavior_rescue_signal", "critic_need_score", "plain_english_reason",
        ]
        st.dataframe(compact_cols(decisions_df, signal_cols), use_container_width=True, hide_index=True)
        st.markdown("#### Governed Summary")
        st.dataframe(governance_summary_df, use_container_width=True, hide_index=True)
    with tech_slate:
        st.dataframe(slate_df, use_container_width=True, hide_index=True)
    with tech_eval:
        st.dataframe(scalable_summary_df, use_container_width=True, hide_index=True)
        with st.expander("Evaluation rows", expanded=False):
            st.dataframe(scalable_eval_df, use_container_width=True, hide_index=True)
    with tech_raw:
        st.dataframe(trace_df, use_container_width=True, hide_index=True)
        raw_output = st.session_state.get("cortex_demo_raw_output", "")
        if raw_output:
            st.code(raw_output)
        elif not trace_df.empty and "governance_output_preview" in trace_df.columns:
            st.code(str(trace_df.iloc[-1].get("governance_output_preview", "")))


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
# Router dry-run logging
# =============================================================================

def load_router_dry_run_log():
    return load_csv_if_exists(ROUTER_DRY_RUN_LOG_PATH)


def append_router_dry_run_log(row):
    os.makedirs(os.path.dirname(ROUTER_DRY_RUN_LOG_PATH), exist_ok=True)

    log_row_df = pd.DataFrame([row])

    if os.path.exists(ROUTER_DRY_RUN_LOG_PATH):
        existing_df = pd.read_csv(ROUTER_DRY_RUN_LOG_PATH)
        output_df = pd.concat([existing_df, log_row_df], ignore_index=True)
    else:
        output_df = log_row_df

    output_df.to_csv(ROUTER_DRY_RUN_LOG_PATH, index=False)


def build_router_log_row(
    query,
    retrieval_mode,
    contract_mode,
    policy_mode,
    selected_policy,
    router_result,
    baseline_results,
    feedback_results,
    slate_reward,
):
    probability_map = router_result.get("probability_map", {})
    feature_row = router_result.get("feature_row", {})

    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "query": query,
        "retrieval_mode": retrieval_mode,
        "contract_mode": contract_mode,
        "policy_mode": policy_mode,
        "selected_policy": selected_policy,
        "router_predicted_policy": router_result.get("predicted_policy", ""),
        "router_confidence": router_result.get("confidence", 0.0),
        "router_prob_baseline": probability_map.get("baseline", 0.0),
        "router_prob_gated_cortex": probability_map.get("gated_cortex", 0.0),
        "router_prob_full_cortex": probability_map.get("full_cortex", 0.0),
        "governance_route": feature_row.get("governance_route", ""),
        "critic_priority": feature_row.get("critic_priority", ""),
        "critic_risk_score": feature_row.get("critic_risk_score", 0.0),
        "gate_final_gate_score": feature_row.get("gate_final_gate_score", 0.0),
        "gate_contract_alignment_score": feature_row.get("gate_contract_alignment_score", 0.0),
        "gate_baseline_confidence_score": feature_row.get("gate_baseline_confidence_score", 0.0),
        "gate_label_quality_score": feature_row.get("gate_label_quality_score", 0.0),
        "gate_exclusion_violation_rate": feature_row.get("gate_exclusion_violation_rate", 0.0),
        "positive_contract_rows": feature_row.get("positive_contract_rows", 0),
        "blocked_rows": feature_row.get("blocked_rows", 0),
        "low_coverage": feature_row.get("low_coverage", 0),
        "top_baseline_product": get_top_product_title(baseline_results),
        "top_cortex_product": get_top_product_title(feedback_results),
        "slate_reward_at_5": slate_reward,
    }


# =============================================================================
# Router dry-run helpers
# =============================================================================

def compute_label_quality_score(df, top_k=5):
    if df is None or len(df) == 0 or "esci_label" not in df.columns:
        return 0.5

    label_scores = {
        "E": 1.0,
        "S": 0.75,
        "C": 0.55,
        "I": 0.0,
        "exact": 1.0,
        "substitute": 0.75,
        "complement": 0.55,
        "irrelevant": 0.0,
    }

    labels = df.head(top_k)["esci_label"].astype(str).tolist()
    scores = [label_scores.get(label, label_scores.get(label.upper(), 0.5)) for label in labels]

    if not scores:
        return 0.5

    return float(sum(scores) / len(scores))


def compute_top_score_metrics(df, score_column):
    if df is None or len(df) == 0:
        return {
            "top1_score": 0.0,
            "top5_mean_score": 0.0,
        }

    selected_score_col = None

    for candidate_col in [score_column, "baseline_score", "semantic_score", "retrieval_score"]:
        if candidate_col in df.columns:
            selected_score_col = candidate_col
            break

    if selected_score_col is None:
        return {
            "top1_score": 0.0,
            "top5_mean_score": 0.0,
        }

    scores = pd.to_numeric(df[selected_score_col], errors="coerce").fillna(0.0)
    top5 = scores.head(5)

    return {
        "top1_score": float(scores.iloc[0]) if len(scores) > 0 else 0.0,
        "top5_mean_score": float(top5.mean()) if len(top5) > 0 else 0.0,
    }


def infer_router_priority(critic_risk_score):
    if critic_risk_score >= 0.75:
        return "critical"
    if critic_risk_score >= 0.55:
        return "high"
    if critic_risk_score >= 0.30:
        return "medium"
    return "low"


def infer_governance_route(critic_priority, query_text, low_coverage):
    query_tokens = str(query_text).split()

    mission_keywords = [
        "party",
        "trip",
        "vacation",
        "camping",
        "wedding",
        "birthday",
        "world cup",
        "watch party",
        "setup",
        "kit",
        "bundle",
    ]

    query_lower = str(query_text).lower()

    if any(keyword in query_lower for keyword in mission_keywords):
        return "mission_candidate_route"

    if low_coverage:
        return "critic_needed_route"

    if critic_priority in ["critical", "high"]:
        return "critic_needed_route"

    if len(query_tokens) <= 2:
        return "full_cortex_route"

    return "full_cortex_route"


def build_live_router_feature_row(
    query,
    baseline_results,
    ranking_input_results,
    contract_enforced_results,
    contract,
    enforcement_report,
    selected_policy,
    score_column,
):
    top_score_metrics = compute_top_score_metrics(baseline_results, score_column)

    if ranking_input_results is not None and len(ranking_input_results) > 0:
        if "contract_match_score" in ranking_input_results.columns:
            contract_alignment_score = float(
                pd.to_numeric(
                    ranking_input_results.head(5)["contract_match_score"],
                    errors="coerce",
                ).fillna(0.0).mean()
            )
        else:
            contract_alignment_score = 0.5
    else:
        contract_alignment_score = 0.0

    if contract_enforced_results is not None and len(contract_enforced_results) > 0:
        if "is_blocked_by_contract" in contract_enforced_results.columns:
            blocked_rows = int(contract_enforced_results["is_blocked_by_contract"].fillna(False).astype(bool).sum())
        else:
            blocked_rows = int(enforcement_report.get("blocked_rows", 0)) if enforcement_report else 0
    else:
        blocked_rows = 0

    positive_contract_rows = (
        int(enforcement_report.get("positive_contract_rows", 0))
        if enforcement_report
        else int(len(ranking_input_results))
    )

    low_coverage = bool(enforcement_report.get("low_coverage", False)) if enforcement_report else False

    candidate_count = max(len(baseline_results), 1)
    exclusion_violation_rate = min(1.0, blocked_rows / candidate_count)

    label_quality_score = compute_label_quality_score(contract_enforced_results, top_k=5)

    baseline_confidence = min(
        1.0,
        max(
            0.0,
            0.55 * top_score_metrics["top1_score"]
            + 0.45 * top_score_metrics["top5_mean_score"],
        ),
    )

    final_gate_score = min(
        1.0,
        max(
            0.0,
            0.40 * baseline_confidence
            + 0.35 * contract_alignment_score
            + 0.25 * label_quality_score
            - 0.15 * exclusion_violation_rate,
        ),
    )

    critic_risk_score = min(
        1.0,
        max(
            0.0,
            1.0
            - (
                0.35 * baseline_confidence
                + 0.35 * contract_alignment_score
                + 0.30 * label_quality_score
            )
            + 0.20 * exclusion_violation_rate
            + (0.15 if low_coverage else 0.0),
        ),
    )

    critic_priority = infer_router_priority(critic_risk_score)
    governance_route = infer_governance_route(critic_priority, query, low_coverage)

    if final_gate_score >= 0.80 and contract_alignment_score >= 0.75:
        gate_decision = "light_rerank"
    else:
        gate_decision = "full_cortex"

    selected_rl_action = selected_policy if selected_policy is not None else "unknown"

    enforcement_status = (
        str(enforcement_report.get("enforcement_status", "unknown"))
        if enforcement_report
        else "unknown"
    )

    row = {
        "critic_risk_score": critic_risk_score,
        "gate_baseline_confidence_score": baseline_confidence,
        "gate_contract_alignment_score": contract_alignment_score,
        "gate_final_gate_score": final_gate_score,
        "gate_top1_score": top_score_metrics["top1_score"],
        "gate_top5_mean_score": top_score_metrics["top5_mean_score"],
        "gate_label_quality_score": label_quality_score,
        "gate_exclusion_violation_rate": exclusion_violation_rate,
        "query_token_count": len(str(query).split()),
        "positive_contract_rows": positive_contract_rows,
        "blocked_rows": blocked_rows,
        "low_coverage": int(low_coverage),
        "needs_online_critic": int(critic_priority in ["critical", "high"]),
        "baseline_loss_rescued": 0,
        "over_rerank_prevented": 0,
        "critic_priority": critic_priority,
        "governance_route": governance_route,
        "gate_decision": gate_decision,
        "selected_rl_action": str(selected_rl_action),
        "enforcement_status": enforcement_status,
    }

    return pd.DataFrame([row])


def build_router_feature_frame_from_artifact(df, feature_payload):
    out = df.copy()

    expected_features = feature_payload.get("feature_columns", [])

    base_numeric = (
        feature_payload.get("base_feature_columns", [])
        + feature_payload.get("optional_numeric_feature_columns", [])
    )

    bool_cols = feature_payload.get("optional_bool_feature_columns", [])
    categorical_cols = feature_payload.get("optional_categorical_columns", [])

    for col in base_numeric:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = safe_numeric_series(out[col])

    for col in bool_cols:
        if col not in out.columns:
            out[col] = 0
        out[col] = out[col].astype(str).str.lower().isin(["true", "1", "yes"]).astype(int)

    existing_categorical_cols = [
        col for col in categorical_cols
        if col in out.columns
    ]

    if existing_categorical_cols:
        cat_df = pd.get_dummies(
            out[existing_categorical_cols].fillna("unknown").astype(str),
            prefix=existing_categorical_cols,
            dummy_na=False,
        )

        out = pd.concat([out, cat_df], axis=1)

    for feature in expected_features:
        if feature not in out.columns:
            out[feature] = 0.0
        out[feature] = safe_numeric_series(out[feature])

    return out, expected_features


def run_live_router_dry_run(
    query,
    baseline_results,
    ranking_input_results,
    contract_enforced_results,
    contract,
    enforcement_report,
    selected_policy,
    score_column,
):
    model = load_router_model()
    feature_payload = load_router_feature_payload()
    metadata = load_router_metadata()

    if model is None or not feature_payload:
        return {
            "available": False,
            "error": "Saved router model or feature schema not found.",
        }

    live_feature_row = build_live_router_feature_row(
        query=query,
        baseline_results=baseline_results,
        ranking_input_results=ranking_input_results,
        contract_enforced_results=contract_enforced_results,
        contract=contract,
        enforcement_report=enforcement_report,
        selected_policy=selected_policy,
        score_column=score_column,
    )

    feature_df, feature_cols = build_router_feature_frame_from_artifact(
        live_feature_row,
        feature_payload,
    )

    predicted_policy = model.predict(feature_df[feature_cols])[0]
    probabilities = model.predict_proba(feature_df[feature_cols])[0]
    classes = list(model.classes_)

    probability_map = {
        str(class_name): float(probabilities[index])
        for index, class_name in enumerate(classes)
    }

    confidence = max(probability_map.values()) if probability_map else 0.0

    return {
        "available": True,
        "predicted_policy": str(predicted_policy),
        "confidence": float(confidence),
        "probability_map": probability_map,
        "feature_row": live_feature_row.to_dict(orient="records")[0],
        "metadata": metadata,
    }


def render_router_dry_run_panel(
    router_result,
    query,
    retrieval_mode,
    contract_mode,
    policy_mode,
    selected_policy,
    baseline_results,
    feedback_results,
    slate_reward,
):
    st.markdown("### Router-Integrated CORTEX Dry Run")

    if not router_result.get("available", False):
        st.warning(router_result.get("error", "Router dry-run unavailable."))
        return

    predicted_policy = router_result.get("predicted_policy", "unknown")
    confidence = router_result.get("confidence", 0.0)
    probability_map = router_result.get("probability_map", {})
    feature_row = router_result.get("feature_row", {})

    st.markdown(
        """
        <div class="info-card-purple">
            Dry-run mode: the saved learned repair router is being used as an advisory layer.
            It predicts which policy should be trusted for this query, but the live slate is still
            produced by the current CORTEX pipeline.
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Router Choice", predicted_policy)
    c2.metric("Router Confidence", f"{confidence * 100:.1f}%")
    c3.metric("Governance Route", feature_row.get("governance_route", "unknown"))
    c4.metric("Critic Priority", feature_row.get("critic_priority", "unknown"))

    prob_df = pd.DataFrame(
        {
            "Policy": list(probability_map.keys()),
            "Probability": list(probability_map.values()),
        }
    )

    if len(prob_df) > 0:
        fig = px.bar(
            prob_df,
            x="Policy",
            y="Probability",
            text="Probability",
            color="Policy",
            title="Router Policy Probability",
            color_discrete_sequence=["#94a3b8", "#10b981", "#3b82f6"],
        )

        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig.update_layout(
            template="plotly_white",
            height=340,
            showlegend=False,
            margin=dict(l=20, r=20, t=60, b=40),
        )

        st.plotly_chart(fig, use_container_width=True)

    if st.button("Save router dry-run decision", use_container_width=True):
        log_row = build_router_log_row(
            query=query,
            retrieval_mode=retrieval_mode,
            contract_mode=contract_mode,
            policy_mode=policy_mode,
            selected_policy=selected_policy,
            router_result=router_result,
            baseline_results=baseline_results,
            feedback_results=feedback_results,
            slate_reward=slate_reward,
        )

        append_router_dry_run_log(log_row)
        st.success(f"Router dry-run decision saved to {ROUTER_DRY_RUN_LOG_PATH}")

    with st.expander("Router Feature Row Used for Dry Run", expanded=False):
        st.json(feature_row)

    with st.expander("Saved Router Metadata", expanded=False):
        st.json(router_result.get("metadata", {}))


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

def render_search_hero():
    st.markdown(
        """
        <div class="search-hero">
            <div class="section-kicker">Live Search Console</div>
            <div class="search-hero-title">Search with governed intelligence.</div>
            <div class="search-hero-subtitle">
                CORTEX retrieves products, understands shopping intent, and shapes a useful
                final slate with guardrails and explainable ranking decisions.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dark_metric_card(label, value, help_text):
    _render_demo_metric(label, value, help_text)


def _contract_intent_text(contract):
    if not isinstance(contract, dict):
        return ""
    for key in ["intent", "query_intent", "shopping_intent", "intent_type"]:
        value = contract.get(key)
        if value:
            return str(value)
    return ""


def build_simple_live_search_explanation(
    selected_policy,
    retrieval_confidence,
    retrieved_count,
    filtered_count,
    enforcement_report,
    contract,
    top_baseline,
    top_final,
    slate_quality,
):
    intent = _contract_intent_text(contract)
    coverage_text = ""
    if enforcement_report and enforcement_report.get("low_coverage", False):
        coverage_text = " Candidate coverage was limited, so contract guardrails remained conservative."
    intent_text = f" for the detected {intent} intent" if intent else ""
    movement_text = (
        f" The top result changed from {top_baseline} to {top_final}."
        if top_baseline and top_final and top_baseline != top_final
        else " The strongest retrieved product remained at the top after governance."
    )
    quality_text = ""
    if slate_quality and "diversity_at_k" in slate_quality:
        quality_text = (
            f" Multi-agent diversity@5 measured {safe_float(slate_quality['diversity_at_k']):.3f}."
        )
    return (
        f"CORTEX used {selected_policy}{intent_text}. Retrieval confidence was "
        f"{retrieval_confidence:.3f}; {filtered_count} of {retrieved_count} retrieved "
        f"candidates informed the ranked slate.{coverage_text}{movement_text}{quality_text}"
    )


def build_business_value_text(query, enforcement_report, category_count, result_count):
    if not result_count or (enforcement_report and enforcement_report.get("low_coverage", False)):
        return (
            "The local ESCI sample has limited matching coverage for this query. CORTEX can "
            "still determine a route, while production results would improve with broader inventory."
        )
    mission_terms = ["setup", "packing", "decorations", "party", "list", "vacation", "shower"]
    is_mission_query = any(term in query.lower() for term in mission_terms) or category_count > 1
    if is_mission_query:
        return (
            "This looks like a broader shopping mission. CORTEX preserves useful coverage "
            "across needs instead of collapsing the result slate around one narrow signal."
        )
    return (
        "This looks like a focused product search. CORTEX keeps relevance central and avoids "
        "unnecessary expansion that could add ranking risk."
    )


def prepare_simple_live_search_display(results):
    if results is None or results.empty:
        return []
    display_rows = []
    for rank, row in enumerate(results.head(6).to_dict(orient="records"), start=1):
        display_rows.append(
            {
                "rank": rank,
                "title": row.get("product_title", "Untitled product"),
                "category": row.get("category", "Relevant match"),
                "brand": row.get("brand", ""),
                "price": row.get("price", ""),
                "reason": row.get("final_slate_enforcement_reason", row.get("contract_filter_reason", "")),
            }
        )
    return display_rows


def render_product_result_cards(results):
    for row in prepare_simple_live_search_display(results):
        price = f" | Price: {row['price']}" if str(row["price"]).strip() not in ["", "nan"] else ""
        reason = (
            f"<br><b>Why included:</b> {escape(str(row['reason']))}"
            if str(row["reason"]).strip() not in ["", "nan"]
            else ""
        )
        st.markdown(
            f"""
            <div class="product-card">
                <div class="product-rank">Rank {row["rank"]}</div>
                <div class="product-title">{escape(str(row["title"]))}</div>
                <div class="product-meta">
                    {escape(str(row["category"]))} | {escape(str(row["brand"]))}{escape(price)}
                    {reason}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_search_console(products):
    render_search_hero()

    mode_col = st.columns([1.4, 1, 1.4])[1]
    with mode_col:
        live_mode = st.radio(
            "Experience mode",
            ["Simple Mode", "Technical Mode"],
            horizontal=True,
            index=0,
            key="live_search_mode",
        )

    search_col = st.columns([1, 2.35, 1])[1]
    with search_col:
        query_input = st.text_input(
            "Shopping query",
            placeholder="Try any shopping query...",
            key="live_search_query",
            label_visibility="collapsed",
        ).strip()
        run_col = st.columns([1, 1.08, 1])[1]
        with run_col:
            run_clicked = st.button(
                "Run CORTEX",
                type="primary",
                use_container_width=True,
                key="run_live_search",
            )

    st.markdown(
        """
        <div class="chip-row">
            <span class="query-chip">beach vacation packing list</span>
            <span class="query-chip">new apartment kitchen setup</span>
            <span class="query-chip">adidas soccer cleats</span>
            <span class="query-chip">office desk setup</span>
            <span class="query-chip">baby shower decorations</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if run_clicked:
        if not query_input:
            st.warning("Enter a shopping query before running CORTEX.")
            return
        st.session_state["last_live_search_query"] = query_input

    query = st.session_state.get("last_live_search_query", "")
    if not query:
        st.markdown(
            """
            <div class="info-card-blue">
                Enter any shopping query and select <b>Run CORTEX</b>. Simple Mode uses the
                recommended governed ranking path automatically.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    if live_mode == "Simple Mode":
        retrieval_mode = "Semantic"
        contract_mode = "LLM Agent"
        policy_mode = "Slate Q-Learning + Multi-Agent Diversification"
        manual_preference = "Most relevant"
    else:
        st.markdown("### Technical Controls")
        top_controls = st.columns([1, 1, 1.65, 1])
        with top_controls[0]:
            retrieval_mode = st.selectbox(
                "Retrieval mode",
                ["Semantic", "TF-IDF"],
            )

        with top_controls[1]:
            contract_mode = st.radio(
                "Contract mode",
                [
                    "LLM Agent",
                    "Rule-based",
                ],
            )

        with top_controls[2]:
            policy_mode = st.radio(
                "Policy mode",
                [
                    "Slate Q-Learning + Multi-Agent Diversification",
                    "Router-Integrated CORTEX Dry Run",
                    "Slate Q-Learning",
                    "Bandit auto-select",
                    "Manual",
                ],
            )

        with top_controls[3]:
            manual_preference = st.selectbox(
                "Manual objective",
                ["Most relevant", "Best rating", "Lowest price"],
            )

    st.markdown(f"### Results for: `{query}`")

    router_dry_run_enabled = policy_mode == "Router-Integrated CORTEX Dry Run"

    effective_policy_mode = (
        "Slate Q-Learning + Multi-Agent Diversification"
        if router_dry_run_enabled
        else policy_mode
    )

    if router_dry_run_enabled:
        st.markdown(
            """
            <div class="info-card-purple">
                Router dry-run is enabled. The app will still run the current best CORTEX pipeline,
                then ask the saved learned router which strategy it would recommend for this query state.
            </div>
            """,
            unsafe_allow_html=True,
        )

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

        if effective_policy_mode == "Bandit auto-select":
            selected_policy = select_policy_epsilon_greedy(feedback_log, epsilon=0.2)

        elif effective_policy_mode in [
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
        filtered_candidate_count = (
            len(contract_filtered_results) if contract_filtered_results is not None else 0
        )

        if contract_filtered_results is not None and len(contract_filtered_results) > 0:
            ranking_input_results = contract_filtered_results
        else:
            ranking_input_results = baseline_results

        if effective_policy_mode == "Bandit auto-select":
            policy_results = apply_policy_rank(ranking_input_results, selected_policy)

        elif effective_policy_mode in [
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
            if effective_policy_mode == "Slate Q-Learning + Multi-Agent Diversification":
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
        st.warning(
            "CORTEX could not find enough matching products in the local ESCI sample for this "
            "query. The query can still be routed, but product results depend on candidate coverage."
        )
        return

    if live_mode == "Simple Mode":
        top_baseline = get_top_product_title(baseline_results)
        top_final = get_top_product_title(feedback_results)
        categories = (
            feedback_results["category"].dropna().astype(str).nunique()
            if "category" in feedback_results.columns
            else 0
        )
        explanation = build_simple_live_search_explanation(
            selected_policy=selected_policy,
            retrieval_confidence=retrieval_confidence,
            retrieved_count=len(baseline_results),
            filtered_count=filtered_candidate_count,
            enforcement_report=enforcement_report,
            contract=contract,
            top_baseline=top_baseline,
            top_final=top_final,
            slate_quality=slate_quality,
        )
        business_text = build_business_value_text(
            query=query,
            enforcement_report=enforcement_report,
            category_count=categories,
            result_count=len(feedback_results),
        )
        st.markdown(
            f"""
            <div class="decision-card">
                <div class="decision-label">CORTEX Result Summary</div>
                <div class="decision-title">{escape(top_final or "A governed product slate is ready.")}</div>
                <div class="small-muted">Top ranked result for <b>{escape(query)}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            render_dark_metric_card("Final Slate Size", len(feedback_results), "Ranked products")
        with c2:
            render_dark_metric_card("Unique Categories", categories, "Coverage in slate")
        with c3:
            render_dark_metric_card("Slate Reward@5", round(slate_reward, 4), "Governed quality")
        with c4:
            render_dark_metric_card("Top Result", top_final[:34] if top_final else "-", "Leading match")

        why_col, business_col = st.columns(2)
        with why_col:
            st.markdown(
                f'<div class="simple-card"><div class="simple-card-title">Why CORTEX chose this ranking route</div>'
                f'<div class="simple-card-body">{escape(explanation)}</div></div>',
                unsafe_allow_html=True,
            )
        with business_col:
            st.markdown(
                f'<div class="simple-card"><div class="simple-card-title">Business meaning</div>'
                f'<div class="simple-card-body">{escape(business_text)}</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("### Final Ranked Slate")
        render_product_result_cards(feedback_results)
        return

    st.markdown("### Technical Run Summary")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Selected Policy", selected_policy)
    c2.metric("Retrieval Confidence", round(retrieval_confidence, 4))
    c3.metric("Retrieved Candidates", len(baseline_results))
    c4.metric("Filtered Candidates", len(ranking_input_results))
    c5.metric("Slate Reward@5", round(slate_reward, 4))

    if router_dry_run_enabled:
        router_result = run_live_router_dry_run(
            query=query,
            baseline_results=baseline_results,
            ranking_input_results=ranking_input_results,
            contract_enforced_results=contract_enforced_results,
            contract=contract,
            enforcement_report=enforcement_report,
            selected_policy=selected_policy,
            score_column=score_column,
        )

        render_router_dry_run_panel(
            router_result=router_result,
            query=query,
            retrieval_mode=retrieval_mode,
            contract_mode=contract_mode,
            policy_mode=policy_mode,
            selected_policy=selected_policy,
            baseline_results=baseline_results,
            feedback_results=feedback_results,
            slate_reward=slate_reward,
        )

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

    if effective_policy_mode in [
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

            if effective_policy_mode in [
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
    router_dry_run_log = load_router_dry_run_log()
    metadata = load_router_metadata()

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

    st.markdown("### Live Router Dry-Run Log")

    if len(router_dry_run_log) > 0:
        log_cols = [
            "timestamp",
            "query",
            "router_predicted_policy",
            "router_confidence",
            "governance_route",
            "critic_priority",
            "critic_risk_score",
            "top_baseline_product",
            "top_cortex_product",
            "slate_reward_at_5",
        ]

        st.dataframe(
            compact_cols(router_dry_run_log.tail(50), log_cols),
            use_container_width=True,
        )

        if "router_predicted_policy" in router_dry_run_log.columns:
            policy_counts = (
                router_dry_run_log["router_predicted_policy"]
                .value_counts()
                .reset_index()
            )
            policy_counts.columns = ["Router Policy", "Count"]

            fig = px.pie(
                policy_counts,
                names="Router Policy",
                values="Count",
                title="Live Dry-Run Router Decisions",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )

            fig.update_layout(template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No live router dry-run decisions saved yet.")

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
            ["MVP 14.4", "Streamlit Router Dry-Run Toggle", "Adds advisory router prediction in the live app."],
            ["MVP 14.5", "Router Decision Logging", "Saves live router dry-run decisions for review."],
            ["MVP 14.6", "Router Dry-Run Analyzer", "Analyzes saved live router decisions."],
            ["MVP 15.1", "Mission Agent", "Detects compound shopping missions."],
            ["MVP 15.2", "Mission Slate Builder", "Builds mission-aware multi-intent slates."],
            ["MVP 15.3", "Mission Guardrails", "Prevents irrelevant mission expansions."],
            ["MVP 15.5", "Mission Coverage Analyzer", "Measures missing sub-intents."],
            ["MVP 15.6", "Mission Critic Agent", "Diagnoses mission slate weaknesses."],
            ["MVP 15.7", "Mission Repair Loop", "Adds missing but useful mission items."],
            ["MVP 15.8", "Repair Quality Guardrails", "Rejects low-quality repairs."],
            ["MVP 15.9", "Strict Compound Repair Rules", "Prevents over-repair on narrow product queries."],
            ["MVP 16", "Behavior-Aware CORTEX", "Adds behavior confidence, mission stage, cold-start rescue, and policy reasons."],
            ["MVP 16.1", "Behavior-Aware Streamlit Page", "Makes MVP 16 interactive."],
            ["MVP 16.2", "Streamlit UI Renovation", "Makes the dashboard product-ready and demo-friendly."],
        ],
        columns=["MVP", "Component", "Purpose"],
    )

    st.dataframe(milestones, use_container_width=True, hide_index=True)

    st.markdown("### Roadmap")

    roadmap = pd.DataFrame(
        [
            ["MVP 19.2C", "Dark Live Search UX", "Present clean decisions while retaining technical traceability."],
            ["MVP 19.3", "Cost vs Value Governance Analyzer", "Measure intervention value against governed execution cost."],
            ["MVP 20", "Final README + Demo Report Polish", "Finish the product narrative and demo evidence."],
        ],
        columns=["Stage", "Planned Capability", "Why It Matters"],
    )

    st.dataframe(roadmap, use_container_width=True, hide_index=True)


# =============================================================================
# Main app
# =============================================================================

products = load_products()

with st.sidebar:
    st.markdown("## CORTEX Engine")

    st.markdown(
        """
        <div class="small-muted">
            <b>Current MVP: 19.2C</b>
            <br><br>
            Simple Mode for demos, Technical Mode for internals.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("### Project State")
    st.write("MVP 19.2C in progress")
    st.write("Governed CORTEX runner complete")
    st.write("Scalable evaluation complete")
    st.write("Dark dual-mode Live Search UX in progress")
    st.write("Cost/value analyzer next")

    st.divider()

    st.markdown("### Useful Command")
    st.code(
        ".\\.venv\\Scripts\\streamlit.exe run app.py",
        language="powershell",
    )

render_header()

tab_search, tab_demo, tab_router, tab_learning, tab_architecture = st.tabs(
    [
        "Live Search Console",
        "CORTEX Demo",
        "Router Dashboard",
        "Learning Dashboard",
        "Architecture",
    ]
)

with tab_search:
    render_search_console(products)

with tab_demo:
    render_cortex_demo()

with tab_router:
    render_router_dashboard()

with tab_learning:
    render_learning_dashboard()

with tab_architecture:
    render_architecture_page()
