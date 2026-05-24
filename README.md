\# CORTEX Engine — Agentic RL Search Ranking System



CORTEX Engine is an experimental search ranking and recommendation research prototype that explores how \*\*agentic AI\*\*, \*\*reinforcement learning\*\*, \*\*contract-aware ranking\*\*, and \*\*learned routing\*\* can improve search result quality beyond a single static reranker.



The project started as a PolicyRank-RL prototype and evolved into a modular agentic ranking system with retrieval, LLM-based intent contracts, candidate filtering, slate-level reinforcement learning, critic verification, repair simulation, and a learned policy router.



\---



\## 1. Project Motivation



Traditional search ranking systems often rely on a fixed pipeline:



```text

query → retrieval → reranker → final ranked list







This approach works well for many queries, but it can fail when:



the query has complex intent,

the first-stage retrieval is already strong and should not be over-reranked,

the query requires diversity or complementary products,

product labels are noisy,

the reranker over-optimizes one signal,

the system needs to decide whether to trust baseline, gated reranking, or full reranking,

different queries require different ranking strategies.



CORTEX is designed around the idea that a ranking system should not always apply one fixed policy. Instead, it should reason about the query, candidate set, contract quality, risk level, and expected reward before choosing the best ranking strategy.



2\. High-Level Idea



CORTEX treats search ranking as a governed multi-agent decision problem.



Instead of using only one ranking function, it uses multiple specialized components:



User Query

&#x20;  ↓

Retrieval Agent

&#x20;  ↓

Contract / Intent Agent

&#x20;  ↓

Contract-Aware Candidate Filter

&#x20;  ↓

Slate Ranking / RL Policy

&#x20;  ↓

Final Slate Enforcement

&#x20;  ↓

Critic / Verifier Agent

&#x20;  ↓

Repair Simulation

&#x20;  ↓

Learned Repair Router

&#x20;  ↓

Router-Integrated CORTEX Policy



The key idea is that the system can choose among:



baseline retrieval

gated CORTEX

full CORTEX



rather than forcing full reranking on every query.



3\. Core Components

3.1 Retrieval Layer



The project supports two retrieval modes:



TF-IDF retrieval

Semantic retrieval



Semantic retrieval is used as the preferred default because it better captures query-product meaning beyond exact token overlap.



Relevant modules:



src/retrieval.py

src/semantic\_retrieval.py

3.2 Search Contract Agent



The contract agent converts a raw query into a structured search contract.



The contract can include:



inferred product type,

required terms,

preferred terms,

blocked terms,

brand preference,

quality preference,

price sensitivity,

ranking weights,

dynamic filters,

explanation.



Two contract modes are supported:



Rule-based contract

LLM Agent contract



Relevant modules:



src/contracts.py

src/llm\_contract\_agent.py

3.3 Contract-Aware Candidate Filtering



After retrieval, CORTEX applies a contract-aware filter. This checks whether candidate products align with the generated search contract.



The filter can score and reason about:



required query terms,

preferred terms,

product type alignment,

brand preference,

blocked terms,

contract match score.



Relevant module:



src/contract\_filters.py

3.4 Final Slate Enforcement



Even after reranking, the final slate can violate user intent. The final slate enforcer applies additional guardrails before showing the result list.



It tracks:



positive contract rows,

blocked rows,

low coverage,

brand preference,

final contract score,

enforcement status,

enforcement explanation.



Relevant module:



src/final\_slate\_enforcer.py

3.5 Slate Q-Learning



CORTEX includes a slate-level reinforcement learning loop. Instead of only ranking individual items, it learns which slate-level action is useful under different contract states.



The Q-learning state can depend on:



contract quality,

retrieval confidence,

query/candidate condition.



The action can change ranking weights across relevance, rating, price, diversity, and contract priority.



Relevant modules:



src/slate\_q\_learning.py

src/policy\_compiler.py

src/slate\_reward.py

3.6 Multi-Agent Diversification



CORTEX includes a multi-agent slate diversification layer that selects positions using multiple agent-like objectives.



The diversifier can reason across:



exact intent matches,

substitutes,

complements,

diversity,

redundancy,

position-level tradeoffs.



Relevant module:



src/multi\_agent\_diversifier.py

3.7 Baseline Preservation Gate



One key lesson from the experiments is that full reranking is not always better. Sometimes the baseline is already strong.



The baseline preservation gate is designed to prevent over-reranking. It allows CORTEX to preserve the baseline, lightly rerank, or fully rerank depending on query and slate conditions.



Relevant module:



src/baseline\_preservation\_gate.py

3.8 Critic / Verifier Agent



The critic agent identifies possible failure modes in the ranking process.



It can flag issues such as:



baseline stronger than CORTEX,

full CORTEX over-reranking,

low contract alignment,

retrieval uncertainty,

possible label noise,

over-diversification,

contract over-filtering,

mission query requiring decomposition.



Relevant modules:



src/critic\_verifier\_agent.py

src/critic\_guided\_repair\_simulator.py

3.9 Learned Repair Router



The learned repair router is one of the most important pieces of the project.



It learns to choose among:



baseline

gated\_cortex

full\_cortex



using features such as:



critic risk score,

critic priority,

governance route,

gate confidence,

contract alignment,

top-k retrieval confidence,

label quality,

exclusion violation rate,

blocked rows,

positive contract rows.



Relevant modules:



src/learned\_repair\_router.py

src/repair\_router\_oos\_validator.py

src/train\_and\_save\_repair\_router.py

src/repair\_router\_inference.py

4\. Current MVP Progress



The project has progressed through several MVP stages.



MVP	Component	Description

MVP 12.x	Experiment logging	Saved query-level experiment snapshots and feedback logs

MVP 13.3	Baseline preservation gate	Prevented unnecessary over-reranking

MVP 13.5	Agent governance controller	Routed queries by risk, cost, and agent need

MVP 13.6	Critic / verifier agent	Diagnosed failure modes and repair actions

MVP 13.7	Critic-guided repair simulator	Simulated repair actions using logged outcomes

MVP 13.8	Learned repair router	Learned policy selection among baseline, gated CORTEX, and full CORTEX

MVP 13.9	Out-of-sample validation	Validated router generalization on held-out data

MVP 14.1	Saved router model	Persisted model, feature schema, and metadata

MVP 14.2	Inference smoke test	Loaded saved router and scored rows

MVP 14.3	Router-integrated scalable evaluator	Created formal router-integrated evaluation reports

MVP 14.4	Streamlit router dry-run toggle	Added advisory router prediction in live app

MVP 14.5	Router decision logging	Logged live router dry-run decisions

MVP 14.6	Router dry-run analyzer	Analyzed saved live router dry-run decisions

5\. Key Experimental Result



The strongest offline evaluation so far used 1000 queries.



The router-integrated policy compared the following strategies:



baseline

gated CORTEX

full CORTEX

router-integrated CORTEX

oracle logged-policy upper bound



Summary:



Strategy	Average Reward@5

Baseline	0.7528

Gated CORTEX	0.8704

Full CORTEX	0.8669

Router-Integrated CORTEX	0.9101

Oracle Upper Bound	0.9162



The learned router achieved:



Router delta vs baseline: +0.1573

Router delta vs gated CORTEX: +0.0397

Router delta vs full CORTEX: +0.0433

Router regret vs oracle: 0.0060

Prediction match rate: 90.0%

Near-oracle rate: 92.4%



This suggests that the router is successfully learning when to trust baseline, gated CORTEX, or full CORTEX.



6\. Streamlit App



The project includes a Streamlit app for interactive experimentation.



The app supports:



live query input,

TF-IDF or semantic retrieval,

rule-based or LLM-generated contracts,

manual policy ranking,

bandit auto-selection,

slate Q-learning,

multi-agent diversification,

router-integrated dry-run mode,

router decision logging,

learning dashboard,

router dashboard,

architecture tab.



Main file:



app.py



Run the app with:



.venv\\Scripts\\streamlit.exe run app.py



or:



.venv\\Scripts\\python.exe -m streamlit run app.py

7\. Main Commands

Run Streamlit app

.venv\\Scripts\\streamlit.exe run app.py

Run scalable evaluator

.venv\\Scripts\\python.exe -m src.scalable\_evaluator --sample-size 1000

Run router-integrated scalable evaluator

.venv\\Scripts\\python.exe -m src.router\_integrated\_scalable\_evaluator

Run router dry-run analyzer

.venv\\Scripts\\python.exe -m src.router\_dry\_run\_analyzer

Check Git status

git status

8\. Important Output Files

Scalable Evaluation

outputs/scalable\_eval\_mvp13\_3\_1\_resilient\_1000.csv

outputs/scalable\_eval\_summary\_mvp13\_3\_1\_resilient\_1000.csv

outputs/scalable\_eval\_by\_query\_mvp13\_3\_1\_resilient\_1000.csv

outputs/scalable\_eval\_lift\_chart\_mvp13\_3\_1\_resilient\_1000.png

Router Evaluation

outputs/router\_integrated\_scalable\_eval.csv

outputs/router\_integrated\_scalable\_eval\_summary.csv

outputs/router\_integrated\_scalable\_eval\_by\_policy.csv

outputs/router\_integrated\_scalable\_eval\_by\_route.csv

outputs/router\_integrated\_scalable\_eval\_high\_impact.csv

Router Dry-Run Logs

storage/router\_dry\_run\_log.csv

outputs/router\_dry\_run\_log\_summary.csv

outputs/router\_dry\_run\_log\_by\_policy.csv

outputs/router\_dry\_run\_log\_by\_route.csv

outputs/router\_dry\_run\_log\_policy\_probability\_audit.csv

Saved Router Model

models/learned\_repair\_router.pkl

models/learned\_repair\_router\_features.json

models/learned\_repair\_router\_metadata.json

9\. Repository Structure

policyrank-rl/

│

├── app.py

├── requirements.txt

├── PROJECT\_STATUS.md

├── README.md

│

├── data/

│   └── esci\_balanced\_sample.csv

│

├── models/

│   ├── learned\_repair\_router.pkl

│   ├── learned\_repair\_router\_features.json

│   ├── learned\_repair\_router\_metadata.json

│   ├── product\_embeddings.pkl

│   └── semantic\_embeddings\_cache.pkl

│

├── outputs/

│   ├── scalable evaluation outputs

│   ├── router evaluation outputs

│   └── dry-run analyzer outputs

│

├── storage/

│   ├── feedback\_log.csv

│   ├── experiment\_runs.csv

│   ├── router\_dry\_run\_log.csv

│   └── slate\_q\_table.json

│

└── src/

&#x20;   ├── retrieval.py

&#x20;   ├── semantic\_retrieval.py

&#x20;   ├── contracts.py

&#x20;   ├── llm\_contract\_agent.py

&#x20;   ├── contract\_filters.py

&#x20;   ├── final\_slate\_enforcer.py

&#x20;   ├── slate\_q\_learning.py

&#x20;   ├── policy\_compiler.py

&#x20;   ├── slate\_reward.py

&#x20;   ├── multi\_agent\_diversifier.py

&#x20;   ├── baseline\_preservation\_gate.py

&#x20;   ├── agent\_governance\_controller.py

&#x20;   ├── critic\_verifier\_agent.py

&#x20;   ├── critic\_guided\_repair\_simulator.py

&#x20;   ├── learned\_repair\_router.py

&#x20;   ├── router\_integrated\_scalable\_evaluator.py

&#x20;   ├── repair\_router\_inference.py

&#x20;   └── router\_dry\_run\_analyzer.py

10\. Setup Notes



This repo uses Git LFS for larger files such as model artifacts, embeddings, outputs, and datasets.



After cloning on a new machine:



git lfs install

git clone https://github.com/srinikhilreddyparvath/policyrank-cortex.git

cd policyrank-cortex

git lfs pull



Create and activate a virtual environment:



python -m venv .venv

.venv\\Scripts\\activate

pip install -r requirements.txt



Create a local .env file if needed:



notepad .env



Do not commit .env.



11\. Current Limitations



This is a research prototype, not a production ranking system.



Current limitations include:



uses Amazon ESCI-style data rather than live production data,

simulated user feedback is used for some reward calculations,

live router dry-run features are approximated from current query/session state,

the learned router has been validated offline but not in a real online environment,

mission-based shopping is not yet fully implemented,

click/purchase behavior integration is planned but not complete,

multimodal ranking is planned but not complete.

12\. Planned Next Steps



Near-term roadmap:



MVP 14.6 cleanup

MVP 14.7 README and setup polish

MVP 15 Mission-Based Shopping Agent

MVP 16 Behavior-Aware CORTEX

MVP 17 Multimodal CORTEX

MVP 18 Online learning loop



Future improvements:



refactor Streamlit app into smaller modules,

move repeated router feature logic into shared utilities,

add tests,

add mission query decomposition,

add bundle-aware ranking,

add click/purchase reward modeling,

add multimodal product understanding,

add safer model governance reports.

13\. Disclaimer



This repository is an independent research and prototype project. It is intended for experimentation, learning, and demonstration of agentic ranking concepts. It is not a production system and should not be treated as production-ready without additional validation, testing, monitoring, and governance.

