# Ticket Categorisation & Triaging using NLP/ML
Reference implementation for the mid-sem architecture diagram (Section 5, Proposed System Architecture):

```
Ticket Inflow → Data Cleaning → Feature Engineering → Orchestrator Agent →
Open/Closed Model Selection → Stage-1 Request Classification → Stage-2 Team
Assignment → Explainability Engine → Confidence Filter → Human Review /
Automated Routing → Feedback Loop → Active Learning → REST API Output
```

This is a **working, runnable skeleton** — trained and smoke-tested end-to-end
on synthetic data — that you can point at the real Infosys ITSM export by
swapping one function (`train.load_dataset`). Everything else (cleaning,
embeddings, hierarchical classifiers, confidence routing, explainability,
feedback loop, API) stays the same.

## Quick start

```bash
pip install -r requirements.txt          # or just pandas/numpy/scikit-learn/joblib for the core pipeline

# One command, whole diagram end-to-end (recommended -- see below):
python run_pipeline.py

# Or run each stage yourself:
python -m src.train                      # generates synthetic data (first run), trains, saves artifacts/
python -m src.evaluate                   # Section 9 success-criteria metrics
uvicorn src.api:app --reload --port 8000 # REST API (needs fastapi/uvicorn)
```

## `run_pipeline.py` -- the full architecture diagram in one script

```bash
python run_pipeline.py                      # full run: train + route + explain + feedback + API demo
python run_pipeline.py --model mlp          # ANN heads instead of linear
python run_pipeline.py --skip-train         # reuse artifacts/ from a previous run
python run_pipeline.py --n-tickets 500      # size of the "live" batch routed through the orchestrator
```

This single entrypoint walks every node in the diagram in order and prints
what happened at each step, so it doubles as a live demo/viva walkthrough
rather than a silent batch job:

1. **Ticket Inflow + Data Cleaning** -- generates + cleans a small sample so you can see the masked/normalized text
2. **Feature Engineering + Model Development** -- trains Stage-1/Stage-2/Priority via `src/train.py` and saves `artifacts/`
3. **Orchestrator Agent + Open/Closed Model Selection** -- reloads artifacts, wires up the explainer, and auto-detects whether a local Ollama LLM fallback is reachable
4. **Stage-1/Stage-2/Priority routing** -- runs a fresh synthetic "live" batch end-to-end through `process_batch`
5. **Explainability Engine** -- prints the top contributing terms for one example ticket
6. **Confidence Filter / governance metrics** -- auto-route vs. review vs. escalate rates, misrouting-flag rate, P1/P2 human-in-loop %
7. **Feedback Loop** -- simulates a reviewer confirming/correcting the routed tickets and logs it via `src/feedback_loop.py`, then prints the resulting disagreement rate (your continuous-learning KPI)
8. **Active Learning** -- ranks the lowest-confidence live tickets so a human reviewer knows what to look at next
9. **REST API Output** -- exercises `src/api.py`'s real `/triage` and `/health` endpoints via FastAPI's `TestClient` if `fastapi`/`httpx` are installed; otherwise calls the identical orchestrator code path `/triage` uses, so the stage is still demonstrated even without the HTTP stack installed

Artifacts land in `artifacts/` (`*.joblib` models + `feedback_log.jsonl`); rerun
with `--skip-train` to iterate on routing/feedback behavior without retraining.

## Linear (Logistic Regression) vs. ANN (MLP)

Every classifier head (Stage-1 category, Stage-2 resolver group, Priority)
can run as either a linear model or a small feed-forward neural network:

```bash
python -m src.train --model logreg   # default: linear, cheap to explain
python -m src.train --model mlp      # ANN: can learn nonlinear term interactions
```

Tradeoff to report in your dissertation's model-ablation table: the ANN can
pick up patterns Logistic Regression can't (e.g. two terms only meaning
something together), but it loses the free `.coef_`-based explainability —
pair `--model mlp` with the SHAP/LIME backends in `explainability.py`
(`get_explainer(backend="shap")`, which falls back to `KernelExplainer` for
non-linear, non-tree models) rather than `LinearWeightExplainer`, which will
raise a clear error if you try to use it on an MLP.

Try the API:
```bash
curl -X POST localhost:8000/triage -H "Content-Type: application/json" -d '{
  "ticket_id": "TCK-1", "title": "VPN not connecting",
  "description": "Critical, all users affected, production down."
}'
```

## How the code maps to the diagram

| Diagram node                          | File                                  |
|----------------------------------------|----------------------------------------|
| Data Cleaning, Dangerous Feature Removal | `src/preprocessing.py`               |
| Feature Engineering / Transformer Embeddings | `src/feature_extraction.py`     |
| Agentic Orchestrator                   | `src/orchestrator.py`                 |
| Stage-1 Request Classification         | `src/models/stage1_classifier.py`     |
| Stage-2 Team Assignment                | `src/models/stage2_classifier.py`     |
| (Priority / SLA head)                  | `src/models/priority_classifier.py`   |
| Explainability Layer (LIME/SHAP)       | `src/explainability.py`               |
| Confidence Filter / Human Review       | `src/confidence_engine.py`            |
| Open/Closed Model Selection (local LLM tier) | `src/llm_fallback.py`           |
| Feedback Loop / Active Learning        | `src/feedback_loop.py`                |
| REST API Integration                   | `src/api.py`                          |
| Evaluation (Accuracy, Precision, Recall, F1, SLA) | `src/evaluate.py`          |
| Ticket Inflow (synthetic, for now)     | `src/data_generator.py`               |
| **End-to-end runner (all of the above, one command)** | `run_pipeline.py`     |

See `strategy/IMPLEMENTATION_STRATEGY.md` for the phase-by-phase build plan,
model ablations, and how this maps to your "Future Plan" timeline
(Model Training – Jul 2026, System Integration – Jul 2026, Testing &
Evaluation – Aug 2026).

## Swapping in real data

Edit `src/train.py::load_dataset()` to read your ITSM export and produce a
DataFrame with columns: `ticket_id, title, description, category, priority,
resolver_group`. Nothing downstream needs to change.

## Experiment tracking with MLflow (self-hosted, offline)

`train.py` logs params/metrics/artifacts to MLflow automatically if it's
installed (`pip install mlflow`) — no code changes needed, and training
still works fine if it's not installed.

**Important: MLflow here is tracking-only, not serving.** Your FastAPI
service (`src/api.py`) is the only thing that answers `/triage` requests.
MLflow has its own built-in model-serving command (`mlflow models serve`),
but we deliberately don't use it — you keep full control of the API
contract, routing logic, explainability, and the local-LLM fallback tier,
none of which MLflow's generic serving wrapper knows about.

**Zero-setup option** (good enough for solo dissertation work): do nothing.
By default, runs are written to a local `./mlruns` folder. View them with:
```bash
mlflow ui   # opens a local dashboard at http://localhost:5000
```

**Self-hosted tracking server** (if you want a persistent server other
machines/collaborators can log to, still fully offline/on-prem):
```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts \
  --host 0.0.0.0 --port 5000
```
Then point training runs at it:
```bash
export MLFLOW_TRACKING_URI=http://localhost:5000   # or your server's LAN IP
python -m src.train --model mlp
```
Both the backend store (`sqlite:///mlflow.db` — swap for Postgres if you
want multi-user concurrent writes) and the artifact root (`./mlartifacts`
— swap for a network share or self-hosted MinIO if you want it off the
tracking-server box) are plain local/on-prem storage. No cloud account,
no billing, works fully air-gapped.

Compare runs (e.g. `logreg` vs `mlp`, `tfidf` vs `transformer`) side by side
in the MLflow UI's table view — this is your dissertation's model-ablation
evidence, generated automatically instead of hand-copied from stdout.

## Local LLM fallback (offline, no API cost)

For tickets the sklearn heads aren't confident about, the orchestrator can
optionally consult a local LLM via [Ollama](https://ollama.com) before
escalating to a human. This is fully offline after a one-time model
download — no API key, no per-token billing.

```bash
# one-time setup
ollama pull mistral
ollama serve   # runs a local server on localhost:11434

# then just run the API/train scripts as normal -- llm_fallback.py
# auto-detects whether Ollama is reachable and no-ops if it isn't
```

## Swapping in real BERT/GUSE embeddings

```python
from src.feature_extraction import get_embedder
embedder = get_embedder("transformer", model_name="sentence-transformers/all-mpnet-base-v2")
```
(requires `pip install sentence-transformers torch`, and model-download
access on whichever machine runs training)

## Current status vs. dissertation "Progress Achieved" section

- [x] Literature review, gap analysis, architecture design — already done (Sections 7, 5)
- [x] Dataset understanding — `data_generator.py` encodes the schema decided in Section 8
- [x] Model development — working baseline (TF-IDF + hierarchical LogisticRegression), swappable for BERT/XGBoost
- [x] API development — FastAPI skeleton with `/triage`, `/triage/batch`, `/feedback`, `/health`
- [x] Feedback loop / active learning — `src/feedback_loop.py`, wired into `run_pipeline.py`
- [x] End-to-end smoke test — `run_pipeline.py` runs every diagram stage on synthetic data in one command
- [ ] Evaluation on **real** ITSM data — `evaluate.py` is ready, just needs the real dataset
- [ ] Integration with the actual ITSM platform (webhook/plugin) — see Phase 6 in the strategy doc
