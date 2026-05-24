import streamlit as st
import pandas as pd
import plotly.express as px

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
)

ensure_storage_exists()

st.title("PolicyRank-RL: CORTEX Engine")
st.subheader("MVP 12.8: Experiment Logging + Snapshot Checkpoint")

st.write(
    "This version saves query-level experiment snapshots for paper-ready analysis: "
    "query, contract, enforcement report, top products, selected policy, and SlateReward@5."
)

products = pd.read_csv("data/esci_balanced_sample.csv")
feedback_log = load_feedback_log()

top_controls = st.columns([2, 1, 1, 1, 1])

with top_controls[0]:
    query = st.text_input(
        "Enter a shopping search query:",
        placeholder="example: adidas soccer cleats",
    )

with top_controls[1]:
    retrieval_mode = st.selectbox(
        "Retrieval mode:",
        ["TF-IDF", "Semantic"],
    )

with top_controls[2]:
    contract_mode = st.radio(
        "Contract mode:",
        [
            "Rule-based",
            "LLM Agent",
        ],
    )

with top_controls[3]:
    policy_mode = st.radio(
        "Policy mode:",
        [
            "Manual",
            "Bandit auto-select",
            "Slate Q-Learning",
            "Slate Q-Learning + Multi-Agent Diversification",
        ],
    )

with top_controls[4]:
    manual_preference = st.selectbox(
        "Manual objective:",
        ["Most relevant", "Best rating", "Lowest price"],
    )

st.divider()


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
        with st.expander("View LLM Ranking Weights", expanded=True):
            st.json(contract.get("ranking_weights", {}))

    if "dynamic_filters" in contract:
        with st.expander("View Dynamic LLM Contract Filters", expanded=True):
            st.json(contract.get("dynamic_filters", {}))


if query:
    st.success(f"You searched for: {query}")

    with st.spinner("Retrieving products..."):
        if retrieval_mode == "TF-IDF":
            baseline_results = tfidf_search(query, products, top_k=20)
            score_column = "retrieval_score"
            baseline_title = "Baseline TF-IDF Ranking"
        else:
            baseline_results = semantic_search(query, products, top_k=20)
            score_column = "semantic_score"
            baseline_title = "Baseline Semantic Ranking"

    if len(baseline_results) > 0:
        retrieval_confidence = float(baseline_results["baseline_score"].max())
    else:
        retrieval_confidence = 0.0

    selected_rl_action = None
    selected_action_weights = None
    contract_state = None
    agent_trace = None
    slate_quality = None
    enforcement_report = None

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

    st.info(f"Selected policy/action: {selected_policy}")

    st.write("### Agentic Search Contract Summary")
    show_contract_metrics(contract)

    with st.expander("View Full Generated Search Contract", expanded=True):
        st.json(contract)

    st.write("### Contract-Aware Candidate Filtering Summary")

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

        existing_filter_cols = [
            col for col in filter_cols if col in ranking_input_results.columns
        ]

        st.dataframe(
            ranking_input_results[existing_filter_cols],
            use_container_width=True,
        )

    st.write("### Final Slate Contract Enforcement Summary")

    if enforcement_report:
        e_col1, e_col2, e_col3, e_col4, e_col5 = st.columns(5)

        e_col1.metric("Enforcement Status", enforcement_report["enforcement_status"])
        e_col2.metric("Preferred Brand", enforcement_report["preferred_brand"])
        e_col3.metric("Positive Contract Rows", enforcement_report["positive_contract_rows"])
        e_col4.metric("Blocked Rows", enforcement_report["blocked_rows"])
        e_col5.metric("Low Coverage", str(enforcement_report["low_coverage"]))

        if enforcement_report["low_coverage"]:
            st.warning(enforcement_report["warning"])
        else:
            st.success(enforcement_report["warning"])

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

            existing_enforcement_cols = [
                col for col in enforcement_cols if col in contract_enforced_results.columns
            ]

            st.dataframe(
                contract_enforced_results[existing_enforcement_cols],
                use_container_width=True,
            )

    if policy_mode in [
        "Slate Q-Learning",
        "Slate Q-Learning + Multi-Agent Diversification",
    ]:
        rl_col1, rl_col2, rl_col3 = st.columns(3)

        rl_col1.metric("Contract State", contract_state)
        rl_col2.metric("Selected RL Action", selected_rl_action)
        rl_col3.metric("Retrieval Confidence", round(retrieval_confidence, 4))

        with st.expander("View Selected RL Action Weights", expanded=True):
            st.json(selected_action_weights)

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

        action_col1, action_col2 = st.columns(2)

        with action_col1:
            if st.button("Save feedback and update learning"):
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
            if st.button("Save experiment snapshot"):
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

        col1, col2 = st.columns(2)

        with col1:
            st.write(f"### {baseline_title}")

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

            existing_baseline_cols = [
                col for col in baseline_display_cols if col in baseline_results.columns
            ]

            st.dataframe(
                baseline_results[existing_baseline_cols],
                use_container_width=True,
            )

        with col2:
            st.write("### PolicyRank / CORTEX-Reranked Slate")

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

            existing_cols = [
                col for col in preferred_cols if col in feedback_results.columns
            ]

            st.dataframe(
                feedback_results[existing_cols],
                use_container_width=True,
            )

        if agent_trace is not None:
            st.write("### Multi-Agent Position Selection Trace")
            st.dataframe(agent_trace, use_container_width=True)

        st.divider()

        st.write("### Slate Reward Summary")

        summary = summarize_feedback(feedback_results)

        metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)

        metric_col1.metric("Total Results", summary["total_results"])
        metric_col2.metric("Simulated Clicks", summary["total_clicks"])
        metric_col3.metric("Total Reward", round(summary["total_reward"], 2))
        metric_col4.metric("Average Reward", round(summary["avg_reward"], 3))
        metric_col5.metric("Slate Reward@5", round(slate_reward, 4))

        if slate_quality is not None:
            st.write("### Multi-Agent Slate Quality@5")

            q_col1, q_col2, q_col3, q_col4, q_col5 = st.columns(5)

            q_col1.metric("Exact@5", slate_quality["exact_count_at_k"])
            q_col2.metric("Substitute@5", slate_quality["substitute_count_at_k"])
            q_col3.metric("Complement@5", slate_quality["complement_count_at_k"])
            q_col4.metric("Irrelevant@5", slate_quality["irrelevant_count_at_k"])
            q_col5.metric("Diversity@5", round(slate_quality["diversity_at_k"], 4))

        st.divider()

        st.write("### Ranking Explanation")

        top_baseline = baseline_results.iloc[0]
        top_policy = feedback_results.iloc[0]

        st.info(
            f"{baseline_title} top result: {top_baseline['product_title']}"
        )

        st.success(
            f"Final top result using '{selected_policy}': "
            f"{top_policy['product_title']}"
        )

        if contract_mode == "LLM Agent":
            st.write("### CORTEX Agent Explanation")
            st.write(contract.get("explanation", "No explanation provided."))

    else:
        st.warning("No matching products found.")

else:
    st.info("Type a search query to begin.")

st.divider()

st.write("## Learning Dashboard")

feedback_log = load_feedback_log()
policy_summary = summarize_policy_rewards(feedback_log)
q_table_df = get_q_table_as_dataframe()
experiment_runs = load_experiment_runs()

dash_col1, dash_col2 = st.columns(2)

with dash_col1:
    st.write("### Old Bandit Policy Reward Summary")
    st.dataframe(policy_summary, use_container_width=True)

with dash_col2:
    st.write("### Average Reward by Old Policy")

    chart = px.bar(
        policy_summary,
        x="ranking_objective",
        y="avg_reward",
        text="avg_reward",
        title="Historical Average Reward per Policy",
    )

    st.plotly_chart(chart, use_container_width=True)

st.write("### Slate Q-Learning Q-Table")

if len(q_table_df) > 0:
    st.dataframe(q_table_df, use_container_width=True)

    q_chart = px.bar(
        q_table_df,
        x="action",
        y="q_value",
        color="state",
        barmode="group",
        title="Contract-Conditioned Q-values by Action",
    )

    st.plotly_chart(q_chart, use_container_width=True)
else:
    st.info("No Q-learning updates yet. Run Slate Q-Learning mode and save feedback.")

st.write("### Experiment Snapshot Log")

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

    existing_display_cols = [
        col for col in display_cols if col in experiment_runs.columns
    ]

    st.dataframe(
        experiment_runs.tail(30)[existing_display_cols],
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

        st.plotly_chart(exp_chart, use_container_width=True)
else:
    st.info("No experiment snapshots saved yet.")

st.write("### Recent Feedback Log")

if len(feedback_log) > 0:
    st.dataframe(feedback_log.tail(20), use_container_width=True)

    metric_col1, metric_col2, metric_col3 = st.columns(3)

    metric_col1.metric("Logged Rows", len(feedback_log))
    metric_col2.metric("Total Logged Clicks", int(feedback_log["simulated_click"].sum()))
    metric_col3.metric(
        "Total Logged Reward",
        round(float(feedback_log["simulated_reward"].sum()), 2),
    )
else:
    st.info("No feedback has been saved yet.")