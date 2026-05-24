\# PolicyRank-RL: CORTEX Engine — MVP 13.2 Checkpoint



\## Current folder

PolicyRank-CORTEX-MVP13



\## Current MVP

MVP 13.2 completed: Scalable Baseline vs CORTEX Evaluation



\## Dataset

Using `data/esci\_balanced\_sample.csv`



\- Rows: 51,077

\- Unique queries: 41,313



\## Current best configuration



\- Retrieval mode: Semantic

\- Contract mode: LLM Agent

\- Policy mode: Slate Q-Learning + Multi-Agent Diversification



\## Implemented components



\- Larger balanced ESCI sample

\- Semantic retrieval with embedding cache

\- LLM-based intent extraction

\- Dynamic LLM search contracts

\- Dynamic contract filters

\- Contract-aware candidate filtering

\- Final slate contract enforcement

\- Brand preference scoring

\- Contract-priority multi-agent diversification

\- Slate Q-learning

\- Feedback logging

\- Experiment snapshot logging

\- Experiment analyzer

\- Baseline vs CORTEX evaluator

\- Scalable evaluator with resume and contract cache



\## Latest 100-query evaluation



\- Total queries: 100

\- CORTEX wins: 74

\- Baseline wins: 25

\- Ties: 1

\- CORTEX win rate: 74%

\- Average baseline SlateReward@5: 0.762573

\- Average CORTEX SlateReward@5: 0.868706

\- Average absolute lift: +0.106133

\- Average percentage lift: +19.7197%

\- Median absolute lift: +0.091971



\## Current technical issue



CORTEX improves most queries but sometimes over-reranks when the semantic baseline is already strong.



\## Next MVP



MVP 13.3: Baseline Preservation Gate



Goal:

If baseline top-5 already has high confidence/reward, preserve baseline order or use light reranking. If baseline appears weak or intent-misaligned, use full CORTEX reranking.



## MVP 13.3.1 completed: Tuned Baseline Preservation Gate

Implemented a confidence-gated reranking safety layer that decides whether to preserve baseline, lightly rerank, or use full CORTEX.

Initial MVP 13.3 gate was too conservative and reduced performance. MVP 13.3.1 tightened the gate thresholds and made full CORTEX the default path unless baseline confidence and contract alignment are very high.

### Latest 100-query MVP 13.3.1 evaluation

- Total queries: 100
- MVP 13.3.1 gated CORTEX wins: 76
- Baseline wins: 24
- Ties: 0
- MVP 13.3.1 win rate: 76%
- MVP 13.2 full CORTEX wins on same run: 72
- Baseline wins vs full CORTEX: 25
- Full CORTEX ties: 3
- Average baseline SlateReward@5: 0.765488
- Average MVP 13.3.1 SlateReward@5: 0.867718
- Average full CORTEX SlateReward@5: 0.863947
- Average MVP 13.3.1 absolute lift: +0.102231
- Average full CORTEX absolute lift: +0.098460
- Average gate delta vs full CORTEX: +0.003771
- Baseline-loss cases rescued: 9
- Over-rerank prevented: 1

### Gate behavior

- Preserve baseline: 0 queries
- Light rerank: 2 queries
- Full CORTEX: 98 queries

### Conclusion

MVP 13.3.1 improved over MVP 13.2 by making CORTEX baseline-aware without suppressing full CORTEX too aggressively. The gate now acts as a selective safety valve rather than a conservative reranking blocker.




## MVP 13.3.1 validated on 1000-query scalable evaluation

MVP 13.3.1 Resilient Baseline-Aware CORTEX Gate was validated on a 1000-query ESCI scalable evaluation.

### 1000-query results

- Total queries: 1000
- MVP 13.3.1 gated CORTEX wins: 741
- Baseline wins vs MVP 13.3.1: 237
- Ties vs MVP 13.3.1: 22
- MVP 13.3.1 win rate: 74.1%

### Full CORTEX comparison

- MVP 13.2 full CORTEX wins: 734
- Baseline wins vs full CORTEX: 255
- Full CORTEX ties: 11
- Full CORTEX win rate: 73.4%

### Reward comparison

- Average baseline SlateReward@5: 0.752832
- Average MVP 13.3.1 SlateReward@5: 0.870426
- Average full CORTEX SlateReward@5: 0.866860
- Average MVP 13.3.1 absolute lift: +0.117594
- Average full CORTEX absolute lift: +0.114029
- Average MVP 13.3.1 percentage lift: +24.71%
- Average full CORTEX percentage lift: +24.05%
- Median MVP 13.3.1 absolute lift: +0.092409
- Median full CORTEX absolute lift: +0.098540
- Average gate delta vs full CORTEX: +0.003566

### Gate behavior

- Preserve baseline: 0 queries
- Light rerank: 17 queries
- Full CORTEX: 983 queries
- LLM contract count: 1000
- Fallback contract count: 0
- Baseline-loss cases rescued: 105
- Over-rerank prevented: 8

### Conclusion

MVP 13.3.1 is the stable validated CORTEX version. On 1000 queries, the baseline-aware gate improved win rate, reduced baseline-loss cases, and improved average SlateReward@5 compared with full CORTEX. The gate is highly selective, allowing full CORTEX for 98.3% of queries while intervening only when reranking risk is detected.



## MVP 13.5 completed: Offline Agent Governance Controller

Implemented an offline Agent Governance Controller that reads CORTEX evaluation logs and assigns each query to a governed agent route.

### Input

- Evaluation file: outputs/scalable_eval_mvp13_3_1_resilient_1000.csv
- Total queries: 1000

### Governance routes

- critic_needed_route: 596 queries
- full_cortex_route: 361 queries
- mission_candidate_route: 30 queries
- light_rerank_route: 13 queries
- baseline_safe_route: 0 queries
- fallback_contract_route: 0 queries

### Overall governance summary

- Average baseline SlateReward@5: 0.752832
- Average gated CORTEX SlateReward@5: 0.870426
- Average full CORTEX SlateReward@5: 0.866860
- Average gated absolute lift: +0.117594
- Average full CORTEX absolute lift: +0.114029
- Average gate delta vs full CORTEX: +0.003566
- Average route cost proxy: 3.643
- Average route reward per cost: 0.246007
- Average route lift per cost: 0.032671

### Route-level findings

#### critic_needed_route

- Query count: 596
- Average gated reward: 0.857314
- Average full CORTEX reward: 0.854295
- Average gate delta: +0.003019
- Full CORTEX lost to baseline: 144
- Gated CORTEX lost to baseline: 139

This route contains the highest number of risky queries and motivates a future Critic / Verifier Agent.

#### full_cortex_route

- Query count: 361
- Average gated reward: 0.889012
- Average full CORTEX reward: 0.885451
- Average gate delta: +0.003561
- Highest major-route reward-per-cost: 0.296337

This route validates full CORTEX as the efficient default path for normal confident queries.

#### mission_candidate_route

- Query count: 30
- Average gated reward: 0.886370
- Average full CORTEX reward: 0.874027
- Average gate delta: +0.012343

Mission-like queries show promising uplift, but appear sparsely in the ESCI benchmark. Mission-based shopping should be added later as a product-facing extension.

#### light_rerank_route

- Query count: 13
- Average gated reward: 0.918641
- Average full CORTEX reward: 0.910152
- Average gate delta: +0.008490

Light rerank is most useful for high-confidence baseline cases.

### Conclusion

MVP 13.5 establishes the first governance layer for CORTEX. It enables route-level analysis, agent-cost proxies, and reward-per-cost evaluation. The largest discovered route is critic_needed_route, suggesting that the next high-value foundation step is a Critic / Verifier Agent rather than mission-based shopping.


## MVP 13.7 completed: Critic-Guided Repair Simulator

Implemented an offline Critic-Guided Repair Simulator that uses critic recommendations to estimate whether repair actions could improve CORTEX rewards.

### Input

- Critic report: outputs/critic_verifier_report.csv
- Total queries: 1000

### Overall repair simulation

- Directly simulated repairs: 727
- Rerun-required repairs: 273
- Repair improves current gated CORTEX: 446
- Repair hurts current gated CORTEX: 0
- Repair neutral: 36
- Requires rerun: 273

### Reward comparison

- Average baseline SlateReward@5: 0.752832
- Average current gated CORTEX SlateReward@5: 0.870426
- Average full CORTEX SlateReward@5: 0.866860
- Average conservative repair reward: 0.905787
- Average oracle logged-policy reward: 0.916152

### Lift comparison

- Current gated CORTEX lift vs baseline: +0.117594
- Full CORTEX lift vs baseline: +0.114029
- Conservative repair lift vs baseline: +0.152955
- Oracle logged-policy lift vs baseline: +0.163320

### Repair deltas

- Conservative repair delta vs current gated CORTEX: +0.035361
- Conservative repair delta vs full CORTEX: +0.038927
- Oracle delta vs current gated CORTEX: +0.045726
- Oracle delta vs full CORTEX: +0.049292

### Major repair actions

#### use_full_cortex_for_similar_queries

- Count: 446
- Average current gated reward: 0.825103
- Average full CORTEX reward: 0.904388
- Average conservative repair delta vs current gated: +0.079284

This is the largest directly simulated repair opportunity.

#### apply_baseline_aware_gate_before_final_slate

- Count: 190
- Average gated reward: 0.883296
- Average full CORTEX reward: 0.784755
- Average conservative delta vs full CORTEX: +0.098541

This confirms that the baseline-aware gate remains essential for full-CORTEX over-reranking cases.

#### rerun_or_repair_contract

- Count: 102
- Requires rerun

#### relax_contract_filter_or_review_blocked_terms

- Count: 101
- Requires rerun

#### increase_retrieval_depth_or_expand_query

- Count: 55
- Requires rerun

### Conclusion

MVP 13.7 shows that critic-guided repair has large offline reward potential. Conservative repair simulation improves average reward from 0.870426 to 0.905787 and average lift from +0.117594 to +0.152955. This is not yet deployable because some decisions use logged outcomes, but it strongly motivates a learned repair router that predicts the best policy before reward is observed.




## MVP 13.8 completed: Learned Repair Router

Implemented an offline Learned Repair Router that predicts the best logged policy for each query among:

- baseline
- gated CORTEX
- full CORTEX

This reframes repair as a multiclass policy-selection problem rather than a binary gate-help prediction problem.

### Input

- Repair simulation file: outputs/critic_guided_repair_simulation.csv
- Total rows: 1000

### Target distribution

- baseline best: 168 queries
- gated CORTEX best: 458 queries
- full CORTEX best: 374 queries

### Classifier performance

Random Forest Repair Router:

- Accuracy: 85.33%
- Balanced accuracy: 85.37%
- Macro precision: 83.58%
- Macro recall: 85.37%
- Macro F1: 84.14%
- Weighted F1: 85.25%

Logistic Regression Repair Router:

- Accuracy: 87.33%
- Balanced accuracy: 86.74%
- Macro precision: 85.42%
- Macro recall: 86.74%
- Macro F1: 85.94%
- Weighted F1: 87.33%

### Policy simulation

- Current gated CORTEX average reward: 0.870426
- Full CORTEX average reward: 0.866860
- Learned repair router average reward: 0.908188
- Oracle logged-policy upper bound reward: 0.916152

### Reward deltas

Learned repair router:

- Delta vs baseline: +0.155357
- Delta vs full CORTEX: +0.041328
- Delta vs current gated CORTEX: +0.037762

Oracle upper bound:

- Delta vs baseline: +0.163320
- Delta vs full CORTEX: +0.049292
- Delta vs current gated CORTEX: +0.045726

### Router policy selection

Learned router selected:

- baseline: 210 queries
- gated CORTEX: 462 queries
- full CORTEX: 328 queries

Oracle best policy distribution:

- baseline: 168 queries
- gated CORTEX: 458 queries
- full CORTEX: 374 queries

### Top predictive features

- critic_risk_score
- critic_priority_critical
- needs_online_critic
- gate_label_quality_score
- critic_priority_high
- critic_priority_medium
- critic_priority_low
- baseline_loss_rescued
- gate_final_gate_score
- gate_baseline_confidence_score
- gate_top5_mean_score
- gate_contract_alignment_score
- gate_top1_score

### Conclusion

MVP 13.8 is the strongest autonomy result so far. It shows that critic/governance/gate signals can predict the best ranking policy with strong held-out classification performance. The learned repair router substantially improves simulated reward over both current gated CORTEX and full CORTEX and approaches the oracle logged-policy upper bound. This remains offline and should be validated out-of-sample before live integration.



## MVP 14 completed: Router-Integrated CORTEX Evaluator Dry Run

Implemented a dry-run integrated evaluator that compares Baseline, Full CORTEX, Gated CORTEX, Router-Integrated CORTEX, and Oracle logged-policy upper bound.

### Input

- Router validation predictions: outputs/repair_router_oos_validation_predictions.csv
- Total queries: 1000
- Train split: 700
- Test split: 300

### Overall results

- Baseline average Reward@5: 0.752832
- Full CORTEX average Reward@5: 0.866860
- Gated CORTEX average Reward@5: 0.870426
- Router-Integrated CORTEX average Reward@5: 0.908188
- Oracle logged-policy upper bound Reward@5: 0.916152

### Overall lift vs baseline

- Full CORTEX lift: +0.114029
- Gated CORTEX lift: +0.117594
- Router-Integrated CORTEX lift: +0.155357
- Oracle lift: +0.163320

### Router deltas

- Router delta vs Gated CORTEX: +0.037762
- Router delta vs Full CORTEX: +0.041328
- Router regret vs oracle: 0.007964

### Held-out test results

Test split, 300 queries:

- Baseline average Reward@5: 0.758811
- Full CORTEX average Reward@5: 0.863460
- Gated CORTEX average Reward@5: 0.868525
- Router-Integrated CORTEX average Reward@5: 0.904957
- Oracle logged-policy upper bound: 0.913875

Test deltas:

- Router delta vs Gated CORTEX: +0.036432
- Router delta vs Full CORTEX: +0.041497
- Router regret vs oracle: 0.008918
- Test prediction match rate: 85.33%
- Test near-oracle rate: 88.67%

### Router policy selection

Overall router selected:

- baseline: 210 queries
- gated CORTEX: 462 queries
- full CORTEX: 328 queries

Oracle best distribution:

- baseline: 168 queries
- gated CORTEX: 458 queries
- full CORTEX: 374 queries

### By predicted policy

When router selected gated CORTEX:

- Query count: 462
- Average router reward: 0.915417
- Prediction match rate: 93.07%
- Near-oracle rate: 95.89%

When router selected full CORTEX:

- Query count: 328
- Average router reward: 0.908475
- Router delta vs gated: +0.060678
- Prediction match rate: 90.55%

When router selected baseline:

- Query count: 210
- Average router reward: 0.891837
- Router delta vs gated: +0.085048
- Router delta vs full CORTEX: +0.054848

### By governance route

critic_needed_route:

- Query count: 596
- Router reward: 0.894746
- Gated reward: 0.857314
- Full CORTEX reward: 0.854295
- Router delta vs gated: +0.037432
- Router delta vs full CORTEX: +0.040451

full_cortex_route:

- Query count: 361
- Router reward: 0.927742
- Gated reward: 0.889012
- Full CORTEX reward: 0.885451
- Router delta vs gated: +0.038730
- Router delta vs full CORTEX: +0.042291

mission_candidate_route:

- Query count: 30
- Router reward: 0.919582
- Gated reward: 0.886370
- Full CORTEX reward: 0.874027
- Router delta vs gated: +0.033212

light_rerank_route:

- Query count: 13
- Router reward: 0.955194
- Gated reward: 0.918641
- Full CORTEX reward: 0.910152
- Router delta vs gated: +0.036552

### Conclusion

MVP 14 is the turning point where CORTEX moves from a fixed agentic reranking pipeline to a router-integrated autonomous ranking policy system. The learned router chooses among baseline, gated CORTEX, and full CORTEX per query and improves reward over every fixed strategy, including on held-out test queries.



## MVP 14.1 completed: Persisted Learned Repair Router

Trained and saved the Learned Repair Router as reusable model artifacts.

### Saved artifacts

- Model: models/learned_repair_router.pkl
- Feature schema: models/learned_repair_router_features.json
- Metadata: models/learned_repair_router_metadata.json

### Training setup

- Input: outputs/critic_guided_repair_simulation.csv
- Training rows: 1000
- Feature count: 24
- Target column: best_logged_policy

### Target distribution

- baseline: 168
- gated CORTEX: 458
- full CORTEX: 374

### Out-of-sample validation check

Random Forest:

- Accuracy: 85.33%
- Balanced accuracy: 85.37%
- Macro F1: 84.14%
- Weighted F1: 85.25%

Logistic Regression:

- Accuracy: 87.33%
- Balanced accuracy: 86.74%
- Macro F1: 85.94%
- Weighted F1: 87.33%

### Saved Random Forest full-data reward summary

- Baseline average Reward@5: 0.752832
- Gated CORTEX average Reward@5: 0.870426
- Full CORTEX average Reward@5: 0.866860
- Saved router average Reward@5: 0.910128
- Oracle average Reward@5: 0.916152

### Saved router deltas

- Router delta vs baseline: +0.157296
- Router delta vs gated CORTEX: +0.039702
- Router delta vs full CORTEX: +0.043268
- Router regret vs oracle: 0.006024
- Prediction match rate: 90.0%

### Saved router policy selection

- baseline: 215
- gated CORTEX: 453
- full CORTEX: 332

### Conclusion

MVP 14.1 converts the Learned Repair Router from an offline validation artifact into a reusable model component. The router remains dry-run only for now, but it can now be loaded by future evaluator or app integrations to choose among baseline, gated CORTEX, and full CORTEX policies per query.




## MVP 14.3 completed: Router-Integrated Scalable Evaluator Dry Run

Implemented a router-integrated scalable evaluator that loads the persisted Learned Repair Router and produces a formal strategy comparison across Baseline, Gated CORTEX, Full CORTEX, Router-Integrated CORTEX, and Oracle logged-policy upper bound.

### Input

- Model: models/learned_repair_router.pkl
- Feature schema: models/learned_repair_router_features.json
- Metadata: models/learned_repair_router_metadata.json
- Evaluation input: outputs/critic_guided_repair_simulation.csv
- Rows evaluated: 1000

### Output files

- outputs/router_integrated_scalable_eval.csv
- outputs/router_integrated_scalable_eval_summary.csv
- outputs/router_integrated_scalable_eval_by_policy.csv
- outputs/router_integrated_scalable_eval_by_route.csv
- outputs/router_integrated_scalable_eval_by_split.csv
- outputs/router_integrated_scalable_eval_high_impact.csv

### Overall results

- Baseline average Reward@5: 0.752832
- Gated CORTEX average Reward@5: 0.870426
- Full CORTEX average Reward@5: 0.866860
- Router-Integrated CORTEX average Reward@5: 0.910128
- Oracle logged-policy upper bound: 0.916152

### Overall lift vs baseline

- Gated CORTEX lift: +0.117594
- Full CORTEX lift: +0.114029
- Router-Integrated CORTEX lift: +0.157296
- Oracle lift: +0.163320

### Router deltas

- Router delta vs Gated CORTEX: +0.039702
- Router delta vs Full CORTEX: +0.043268
- Router regret vs Oracle: 0.006024
- Prediction match rate: 90.0%
- Near-oracle rate: 92.4%

### Router policy selection

- Predicted baseline: 215
- Predicted gated CORTEX: 453
- Predicted full CORTEX: 332

Oracle distribution:

- Oracle baseline: 168
- Oracle gated CORTEX: 458
- Oracle full CORTEX: 374

### By predicted policy

When router selected gated CORTEX:

- Query count: 453
- Router reward: 0.915804
- Router lift vs baseline: +0.192302
- Prediction match rate: 96.03%
- Near-oracle rate: 98.23%

When router selected full CORTEX:

- Query count: 332
- Router reward: 0.913054
- Router lift vs baseline: +0.211396
- Router delta vs gated: +0.065769
- Prediction match rate: 93.07%

When router selected baseline:

- Query count: 215
- Router reward: 0.893650
- Router delta vs gated: +0.083101
- Router delta vs full CORTEX: +0.056196
- Prediction match rate: 72.56%

### By governance route

critic_needed_route:

- Query count: 596
- Gated reward: 0.857314
- Full CORTEX reward: 0.854295
- Router reward: 0.897724
- Oracle reward: 0.904552
- Router delta vs gated: +0.040410
- Router delta vs full CORTEX: +0.043429

full_cortex_route:

- Query count: 361
- Gated reward: 0.889012
- Full CORTEX reward: 0.885451
- Router reward: 0.928173
- Oracle reward: 0.932010
- Router delta vs gated: +0.039161
- Router delta vs full CORTEX: +0.042723

mission_candidate_route:

- Query count: 30
- Gated reward: 0.886370
- Full CORTEX reward: 0.874027
- Router reward: 0.917903
- Oracle reward: 0.934910
- Router delta vs gated: +0.031533
- Router delta vs full CORTEX: +0.043876

light_rerank_route:

- Query count: 13
- Gated reward: 0.918641
- Full CORTEX reward: 0.910152
- Router reward: 0.959742
- Oracle reward: 0.964290
- Router delta vs gated: +0.041101
- Router delta vs full CORTEX: +0.049590

### Conclusion

MVP 14.3 formalizes the saved Learned Repair Router as a dry-run scalable evaluation component. Router-Integrated CORTEX improves average Reward@5 over both Gated CORTEX and Full CORTEX and remains close to the oracle logged-policy upper bound. This establishes the first reusable router-integrated evaluation layer for CORTEX.
