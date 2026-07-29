"""
run_pipeline.py
-----------------
Single-command, end-to-end run of the full architecture diagram:

    Ticket Inflow -> Data Cleaning -> Feature Engineering -> Orchestrator Agent
    -> Open/Closed Model Selection -> Stage-1 Request Classification
    -> Stage-2 Team Assignment -> Explainability Engine -> Confidence Filter
    -> Human Review / Automated Routing -> Feedback Loop -> Active Learning
    -> REST API Output

Usage:
    python run_pipeline.py                      # full run, logreg heads, tfidf embeddings
    python run_pipeline.py --model mlp          # ANN heads instead of linear
    python run_pipeline.py --n-tickets 500      # smaller synthetic batch for a quick smoke test
    python run_pipeline.py --skip-train         # reuse artifacts/ from a previous run

What this script actually does, stage by stage:
    1.  Generate/load synthetic tickets                 (src/data_generator.py)
    2.  Clean + mask dangerous features                 (src/preprocessing.py)
    3.  Train Stage-1/Stage-2/Priority + save artifacts  (src/train.py)      [skippable]
    4.  Reload artifacts + build the orchestrator        (src/orchestrator.py)
    5.  Route a fresh batch of "live" tickets end-to-end  (process_batch)
    6.  Run the Explainability Engine on a sample ticket  (src/explainability.py)
    7.  Print governance/routing metrics                 (src/evaluate.py logic)
    8.  Simulate human review + log feedback              (src/feedback_loop.py)
    9.  Rank remaining low-confidence tickets for active learning
    10. Exercise the REST API layer directly (FastAPI TestClient if installed,
        otherwise call the same orchestrator methods api.py calls, so the
        "REST API Output" stage is still demonstrated end-to-end)

Every stage prints what it did, so this doubles as a live walkthrough of the
diagram for a viva/demo -- not just a batch job.
"""

import argparse
import sys
import pandas as pd

from src.data_generator import generate_dataset
from src.preprocessing import clean_dataframe
from src.explainability import LinearWeightExplainer
from src.orchestrator import TicketTriageOrchestrator
from src.feedback_loop import FeedbackStore, FeedbackRecord, select_for_active_learning
from src.llm_fallback import get_llm_fallback

ARTIFACT_DIR = "artifacts"


def _banner(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def load_artifacts():
    import joblib
    embedder = joblib.load(f"{ARTIFACT_DIR}/embedder.joblib")
    stage1 = joblib.load(f"{ARTIFACT_DIR}/stage1_category.joblib")
    stage2 = joblib.load(f"{ARTIFACT_DIR}/stage2_resolver.joblib")
    priority = joblib.load(f"{ARTIFACT_DIR}/priority.joblib")
    return embedder, stage1, stage2, priority


def build_orchestrator(embedder, stage1, stage2, priority):
    explainer = None
    try:
        explainer = LinearWeightExplainer(stage1.model, embedder.get_feature_names())
    except Exception as e:
        print(f"    (explainer unavailable for this model type: {e})")

    llm_fallback = get_llm_fallback()
    print(f"    local LLM fallback (Ollama): {'available' if llm_fallback else 'not running, skipping tier 2'}")

    return TicketTriageOrchestrator(
        embedder, stage1, stage2, priority,
        explainer=explainer,
        llm_fallback=llm_fallback,
    )


def stage_train(model_type: str, embedding_backend: str, n_train: int):
    _banner(f"[3] TRAINING  (model={model_type}, embedding={embedding_backend})")
    # Force a fresh synthetic dataset for this run rather than reusing any
    # cached data/synthetic_tickets.csv from a previous invocation.
    import os
    csv_path = "data/synthetic_tickets.csv"
    if os.path.exists(csv_path):
        os.remove(csv_path)

    from src import train as train_module
    train_module.main(embedding_backend=embedding_backend, model_type=model_type)


def stage_route_live_batch(orchestrator, n_tickets: int) -> pd.DataFrame:
    _banner(f"[5] ROUTING A FRESH BATCH OF {n_tickets} 'LIVE' TICKETS END-TO-END")
    live = generate_dataset(n_tickets)
    decisions = orchestrator.process_batch(live)
    results = pd.DataFrame([d.__dict__ for d in decisions])
    truth = live.reset_index(drop=True)

    print(results[["ticket_id", "category", "resolver_group", "priority", "action", "misrouting_flag"]]
          .head(10).to_string(index=False))
    print(f"    ... {len(results)} tickets routed total")
    return results, truth


def stage_explain_one(orchestrator, ticket_id, title, description):
    _banner("[6] EXPLAINABILITY ENGINE -- single ticket walkthrough")
    decision = orchestrator.process_ticket(ticket_id, title, description)
    print(f"    Ticket: '{title}' / '{description}'")
    print(f"    -> category={decision.category} (conf={decision.category_confidence}), "
          f"resolver_group={decision.resolver_group} (conf={decision.resolver_confidence}), "
          f"priority={decision.priority}")
    print(f"    -> action={decision.action} | reason: {decision.reason}")
    if decision.explanation:
        print("    Top contributing terms:")
        for item in decision.explanation[:5]:
            if "term" in item:
                print(f"        {item['term']:<20} contribution={item.get('contribution')}")
    else:
        print("    (no explanation available -- linear explainer requires a logreg head)")
    return decision


def stage_governance_metrics(results: pd.DataFrame, truth: pd.DataFrame):
    _banner("[7] CONFIDENCE FILTER / GOVERNANCE METRICS")
    action_counts = results["action"].value_counts(normalize=True) * 100
    print(action_counts.round(1).to_string())
    print(f"Misrouting flag rate: {results['misrouting_flag'].mean() * 100:.1f}%")
    p1_p2 = results[truth["priority"].isin(["P1-Critical", "P2-High"])]
    if len(p1_p2):
        pct_human_touched = (p1_p2["action"] != "auto_route").mean() * 100
        print(f"P1/P2 tickets with a human in the loop: {pct_human_touched:.1f}% (target: 100%)")


def stage_feedback_and_active_learning(results: pd.DataFrame, truth: pd.DataFrame):
    _banner("[8] FEEDBACK LOOP -- simulating human review on escalated/reviewed tickets")
    store = FeedbackStore(path=f"{ARTIFACT_DIR}/feedback_log.jsonl")

    reviewed = results[results["action"] != "auto_route"].head(15)
    n_logged = 0
    for _, row in reviewed.iterrows():
        truth_row = truth[truth["ticket_id"] == row["ticket_id"]]
        if truth_row.empty:
            continue
        true_resolver = truth_row.iloc[0]["resolver_group"]
        store.log(FeedbackRecord(
            ticket_id=row["ticket_id"],
            field="resolver_group",
            model_prediction=row["resolver_group"],
            model_confidence=row["resolver_confidence"],
            human_label=true_resolver,          # simulate the reviewer confirming/overriding
            reviewer_id="demo_reviewer",
        ))
        n_logged += 1
    print(f"    Logged {n_logged} human-review feedback records to {store.path}")
    rate = store.disagreement_rate(field_name="resolver_group")
    print(f"    Resolver-group disagreement rate so far: {rate * 100:.1f}% "
          f"(this is your continuous-learning KPI -- track it run over run)")

    _banner("[9] ACTIVE LEARNING -- prioritizing the lowest-confidence tickets for review")
    candidates = results.to_dict("records")
    picked = select_for_active_learning(candidates, n=5, confidence_key="resolver_confidence")
    print("    Top 5 tickets a human reviewer should look at next (lowest resolver confidence):")
    for d in picked:
        print(f"        {d['ticket_id']}  resolver_group={d['resolver_group']}  "
              f"confidence={d['resolver_confidence']:.3f}")
    return store


def stage_rest_api_output(orchestrator):
    _banner("[10] REST API OUTPUT")
    try:
        from fastapi.testclient import TestClient
        import src.api as api_module
        # Reuse the already-built orchestrator instead of re-loading artifacts
        # inside the app's startup event, so this exercises the real endpoint
        # code paths without needing a second training pass.
        api_module._state["orchestrator"] = orchestrator
        api_module._state["feedback_store"] = FeedbackStore(path=f"{ARTIFACT_DIR}/feedback_log.jsonl")
        client = TestClient(api_module.app)

        resp = client.post("/triage", json={
            "ticket_id": "TCK-DEMO-1",
            "title": "VPN not connecting",
            "description": "Critical, all users affected, production down.",
        })
        print(f"    POST /triage -> {resp.status_code}")
        print(f"    {resp.json()}")

        health = client.get("/health")
        print(f"    GET /health -> {health.status_code} {health.json()}")
    except ImportError:
        print("    fastapi/httpx not installed in this environment, so the live HTTP")
        print("    layer can't be exercised here. Calling the same orchestrator method")
        print("    api.py's /triage endpoint calls, to demonstrate the identical code path:")
        decision = orchestrator.process_ticket(
            "TCK-DEMO-1", "VPN not connecting",
            "Critical, all users affected, production down.",
        )
        print(f"    -> {decision.__dict__}")
        print("\n    To exercise the real HTTP API: pip install -r requirements.txt "
              "&& uvicorn src.api:app --reload --port 8000")


def main():
    parser = argparse.ArgumentParser(description="Run the full ticket-triage pipeline end-to-end.")
    parser.add_argument("--model", choices=["logreg", "mlp"], default="logreg")
    parser.add_argument("--embedding", choices=["tfidf", "transformer"], default="tfidf")
    parser.add_argument("--n-train", type=int, default=3000, help="synthetic tickets to train on")
    parser.add_argument("--n-tickets", type=int, default=100, help="synthetic 'live' tickets to route in the demo batch")
    parser.add_argument("--skip-train", action="store_true", help="reuse artifacts/ from a previous run instead of retraining")
    args = parser.parse_args()

    _banner("[1-2] TICKET INFLOW + DATA CLEANING (smoke test)")
    sample = generate_dataset(5)
    cleaned = clean_dataframe(sample)
    print(f"    Generated {len(sample)} sample tickets, cleaned -> {len(cleaned)} rows "
          f"(duplicates_removed={cleaned.attrs.get('duplicates_removed', 0)})")
    print(cleaned[["ticket_id", "clean_text"]].head(2).to_string(index=False))

    if args.skip_train:
        _banner("[3] TRAINING -- skipped (--skip-train), reusing existing artifacts/")
    else:
        stage_train(args.model, args.embedding, args.n_train)

    _banner("[4] ORCHESTRATOR -- reloading artifacts + wiring up Stage-1/Stage-2/Priority/Explainer/LLM-fallback")
    embedder, stage1, stage2, priority = load_artifacts()
    orchestrator = build_orchestrator(embedder, stage1, stage2, priority)

    results, truth = stage_route_live_batch(orchestrator, args.n_tickets)

    stage_explain_one(
        orchestrator, "TCK-DEMO-0",
        "Cannot reach internal server",
        "DNS resolution failing intermittently, second time this has occurred this week.",
    )

    stage_governance_metrics(results, truth)
    stage_feedback_and_active_learning(results, truth)
    stage_rest_api_output(orchestrator)

    _banner("PIPELINE COMPLETE")
    print("Artifacts:        artifacts/*.joblib")
    print("Feedback log:     artifacts/feedback_log.jsonl")
    print("Run again with --skip-train to iterate on routing/feedback logic only.")


if __name__ == "__main__":
    sys.exit(main())
