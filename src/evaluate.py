"""
evaluate.py
------------
Implements Section 9 (Success Criteria) measurement:
    - Accuracy, Precision, Recall, F1 for category / resolver group / priority
    - Routing-level metrics: % auto-routed, % escalated, misrouting-flag rate
    - SLA-aware metric: share of P1/P2 tickets escalated for human co-sign
      (these should be ~100% by the confidence-engine's design, so this
      metric is really a policy-conformance check, not a model metric)

Run: python -m src.evaluate
"""

import joblib
import pandas as pd
from sklearn.metrics import classification_report

from src.data_generator import generate_dataset
from src.preprocessing import clean_dataframe
from src.orchestrator import TicketTriageOrchestrator
from src.explainability import LinearWeightExplainer

ARTIFACT_DIR = "artifacts"


def load_artifacts():
    embedder = joblib.load(f"{ARTIFACT_DIR}/embedder.joblib")
    stage1 = joblib.load(f"{ARTIFACT_DIR}/stage1_category.joblib")
    stage2 = joblib.load(f"{ARTIFACT_DIR}/stage2_resolver.joblib")
    priority = joblib.load(f"{ARTIFACT_DIR}/priority.joblib")
    return embedder, stage1, stage2, priority


def main(n_eval=600):
    embedder, stage1, stage2, priority = load_artifacts()

    explainer = None
    try:
        explainer = LinearWeightExplainer(stage1.model, embedder.get_feature_names())
    except Exception:
        pass

    orchestrator = TicketTriageOrchestrator(embedder, stage1, stage2, priority, explainer)

    df = clean_dataframe(generate_dataset(n_eval))
    decisions = orchestrator.process_batch(df)

    results = pd.DataFrame([d.__dict__ for d in decisions])
    truth = df.reset_index(drop=True)

    print("=== Classification metrics ===")
    print("Category:")
    print(classification_report(truth["category"], results["category"], zero_division=0))
    print("Resolver group:")
    print(classification_report(truth["resolver_group"], results["resolver_group"], zero_division=0))
    print("Priority:")
    print(classification_report(truth["priority"], results["priority"], zero_division=0))

    print("=== Routing / governance metrics ===")
    action_counts = results["action"].value_counts(normalize=True) * 100
    print(action_counts.round(1).to_string())
    print(f"Misrouting flag rate: {results['misrouting_flag'].mean() * 100:.1f}%")

    p1_p2 = results[truth["priority"].isin(["P1-Critical", "P2-High"])]
    if len(p1_p2):
        pct_human_touched = (p1_p2["action"] != "auto_route").mean() * 100
        print(f"P1/P2 tickets with a human in the loop: {pct_human_touched:.1f}% (target: 100%)")


if __name__ == "__main__":
    main()
