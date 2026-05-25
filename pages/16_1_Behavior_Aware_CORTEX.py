
"""
MVP 16.1: Behavior-Aware CORTEX Streamlit Page

Adds an interactive Streamlit page for MVP 16 behavior-aware ranking.

This page:
- Takes a shopping query
- Runs src.behavior_aware_cortex
- Displays behavior-aware summary
- Displays ranked slate
- Displays policy reason codes
- Highlights cold-start rescue and exploration items
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


OUTPUT_DIR = Path("outputs")

BEHAVIOR_SLATE_PATH = OUTPUT_DIR / "behavior_aware_cortex_slate.csv"
BEHAVIOR_SUMMARY_PATH = OUTPUT_DIR / "behavior_aware_cortex_summary.csv"
BEHAVIOR_REASONS_PATH = OUTPUT_DIR / "behavior_aware_cortex_policy_reasons.csv"


st.set_page_config(
    page_title="MVP 16.1 Behavior-Aware CORTEX",
    page_icon="🧠",
    layout="wide",
)


st.title("MVP 16.1 — Behavior-Aware CORTEX")
st.caption(
    "Mission-stage scoring + behavior confidence + cold-start rescue + exploration policy + reason codes."
)


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


default_queries = [
    "new apartment kitchen setup",
    "beach vacation packing list",
    "adidas soccer cleats",
    "world cup watch party",
    "camping trip essentials",
]

with st.sidebar:
    st.header("Run MVP 16")
    selected_query = st.selectbox("Example query", default_queries)
    custom_query = st.text_input("Or enter custom query", value=selected_query)

    run_clicked = st.button("Run Behavior-Aware CORTEX", type="primary")

    st.divider()
    st.markdown("### What this page shows")
    st.markdown(
        """
        - Behavior-aware final ranking
        - Mission stage
        - Behavior confidence
        - Cold-start rescue flags
        - Exploration flags
        - Policy reason codes
        """
    )


query = custom_query.strip()

if run_clicked:
    if not query:
        st.warning("Please enter a query.")
    else:
        with st.spinner(f"Running Behavior-Aware CORTEX for: {query}"):
            ok, output = run_behavior_aware_cortex(query)

        if ok:
            st.success("Behavior-Aware CORTEX completed.")
        else:
            st.error("Behavior-Aware CORTEX failed.")

        with st.expander("Raw run output", expanded=False):
            st.code(output)


summary_df = filter_by_query(read_csv_safe(BEHAVIOR_SUMMARY_PATH), query)
slate_df = filter_by_query(read_csv_safe(BEHAVIOR_SLATE_PATH), query)
reasons_df = filter_by_query(read_csv_safe(BEHAVIOR_REASONS_PATH), query)


st.subheader("Behavior-Aware Summary")

if summary_df.empty:
    st.info("No summary found yet. Run the page for a query first.")
else:
    summary_row = summary_df.iloc[-1].to_dict()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Input Strict Slate", summary_row.get("input_strict_slate_size", 0))
    col2.metric("Final Behavior Slate", summary_row.get("final_behavior_slate_size", 0))
    col3.metric("Unique Sub-Intents", summary_row.get("unique_sub_intents", 0))
    col4.metric("Exploration Items", summary_row.get("exploration_items", 0))

    col5, col6, col7, col8 = st.columns(4)

    col5.metric("High Confidence", summary_row.get("high_confidence_items", 0))
    col6.metric("Medium Confidence", summary_row.get("medium_confidence_items", 0))
    col7.metric("Low Confidence", summary_row.get("low_confidence_items", 0))
    col8.metric("Cold-Start Rescue", summary_row.get("cold_start_rescue_items", 0))

    st.markdown("#### Decision")
    st.code(str(summary_row.get("final_decision", "")))

    st.markdown("#### Explanation")
    st.write(summary_row.get("explanation", ""))


st.divider()

st.subheader("Behavior-Aware Ranked Slate")

if slate_df.empty:
    st.info("No slate found yet for this query.")
else:
    display_cols = [
        "behavior_rank",
        "sub_intent",
        "sub_intent_role",
        "mission_stage",
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

    existing_cols = [c for c in display_cols if c in slate_df.columns]

    slate_df = slate_df.sort_values("behavior_rank") if "behavior_rank" in slate_df.columns else slate_df

    st.dataframe(
        slate_df[existing_cols],
        use_container_width=True,
        hide_index=True,
    )

    csv_bytes = slate_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download behavior-aware slate CSV",
        data=csv_bytes,
        file_name="behavior_aware_cortex_slate_filtered.csv",
        mime="text/csv",
    )


st.divider()

st.subheader("Policy Reason Codes")

if reasons_df.empty:
    st.info("No policy reasons found yet for this query.")
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
    )


st.divider()

st.subheader("Policy Reason Distribution")

if slate_df.empty or "policy_reason" not in slate_df.columns:
    st.info("Run a query to see policy reason distribution.")
else:
    reason_counts = (
        slate_df["policy_reason"]
        .value_counts()
        .reset_index()
    )
    reason_counts.columns = ["policy_reason", "count"]

    st.bar_chart(
        reason_counts,
        x="policy_reason",
        y="count",
    )


st.divider()

st.subheader("MVP 16.1 Interpretation")

st.markdown(
    """
    This page turns MVP 16 into an interactive demo.

    The key difference from a normal ranking system is that CORTEX does not only boost high-behavior items.
    It also protects useful mission-completion items through cold-start rescue and exploration policy logic.
    """
)
