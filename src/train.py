"""
train.py
---------
Top-level training entrypoint. Run:

    python -m src.train

This covers Phase 2 (Dataset Collection & Cleaning), Phase 3 (Feature
Engineering), and Phase 4 (Model Development) from the Methodology section
in one script, and saves artifacts that `api.py` loads at request time.

To point this at the real Infosys ITSM export instead of synthetic data,
replace `load_dataset()` with a loader that reads your export and maps its
columns to: ticket_id, title, description, category, priority,
resolver_group. Nothing else in this file needs to change.
"""

import os
import argparse
import pandas as pd

from src.data_generator import generate_dataset
from src.preprocessing import clean_dataframe
from src.feature_extraction import get_embedder
from src.models.stage1_classifier import Stage1CategoryClassifier, train_val_split
from src.models.stage2_classifier import Stage2TeamAssignmentClassifier
from src.models.priority_classifier import PriorityClassifier
from src.mlflow_utils import TrackingRun

ARTIFACT_DIR = "artifacts"


def load_dataset(csv_path="data/synthetic_tickets.csv", n_synthetic=3000):
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    df = generate_dataset(n_synthetic)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    return df


def main(embedding_backend="tfidf", model_type="logreg"):
    """model_type: "logreg" (linear baseline) or "mlp" (small ANN, nonlinear).
    Run both and diff the printed F1/accuracy numbers for your dissertation's
    model-ablation table -- e.g.:
        python -m src.train --model logreg
        python -m src.train --model mlp
    Every run is also logged to MLflow (params + metrics + artifacts) if
    mlflow is installed, so you can compare runs in the MLflow UI instead of
    just reading stdout. See README for how to host the tracking server.
    """
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    with TrackingRun(run_name=f"{model_type}_{embedding_backend}") as run:
        run.log_params({"model_type": model_type, "embedding_backend": embedding_backend})

        print("[1/5] Loading dataset...")
        df = load_dataset()

        print("[2/5] Cleaning (dedupe, missing values, dangerous-feature masking)...")
        df = clean_dataframe(df)
        dupes_removed = df.attrs.get("duplicates_removed", 0)
        print(f"    removed {dupes_removed} duplicate tickets")
        run.log_params({"n_tickets_after_clean": len(df), "duplicates_removed": dupes_removed})

        print("[3/5] Feature engineering...")
        embedder = get_embedder(embedding_backend)
        X_train_text, X_test_text, y_cat_train, y_cat_test, idx_train, idx_test = _split_with_index(df)
        X_train = embedder.fit_transform(X_train_text)
        X_test = embedder.transform(X_test_text)

        print(f"[4/5] Training Stage-1 (category), Stage-2 (resolver group), Priority head [model_type={model_type}]...")
        stage1 = Stage1CategoryClassifier(model_type=model_type).fit(X_train, y_cat_train)
        report1 = stage1.evaluate(X_test, y_cat_test)
        stage1_f1 = report1["macro avg"]["f1-score"]
        print("    Stage-1 category macro F1:", round(stage1_f1, 3))

        stage2 = Stage2TeamAssignmentClassifier().fit(
            X_train, df.loc[idx_train, "category"].values, df.loc[idx_train, "resolver_group"].values
        )
        res_pred, res_conf = stage2.predict_with_confidence(X_test, df.loc[idx_test, "category"].values)
        res_acc = (res_pred == df.loc[idx_test, "resolver_group"].values).mean()
        print("    Stage-2 resolver-group accuracy:", round(res_acc, 3))

        priority = PriorityClassifier(model_type=model_type).fit(X_train, df.loc[idx_train, "priority"].values)
        pri_pred, pri_conf = priority.predict_with_confidence(X_test)
        pri_acc = (pri_pred == df.loc[idx_test, "priority"].values).mean()
        print("    Priority accuracy:", round(pri_acc, 3))

        run.log_metrics({
            "stage1_category_macro_f1": stage1_f1,
            "stage2_resolver_accuracy": res_acc,
            "priority_accuracy": pri_acc,
        })

        print("[5/5] Saving artifacts to", ARTIFACT_DIR)
        import joblib
        joblib.dump(embedder, f"{ARTIFACT_DIR}/embedder.joblib")
        stage1.save(f"{ARTIFACT_DIR}/stage1_category.joblib")
        stage2.save(f"{ARTIFACT_DIR}/stage2_resolver.joblib")
        priority.save(f"{ARTIFACT_DIR}/priority.joblib")

        # Log the artifact files themselves to MLflow too, so a given run is
        # fully self-contained and reproducible from the MLflow UI alone.
        for f in ["embedder.joblib", "stage1_category.joblib", "stage2_resolver.joblib", "priority.joblib"]:
            run.log_artifact(f"{ARTIFACT_DIR}/{f}")

        print("Done.")


def _split_with_index(df, test_size=0.2, random_state=42):
    from sklearn.model_selection import train_test_split
    idx = df.index.values
    idx_train, idx_test = train_test_split(
        idx, test_size=test_size, random_state=random_state, stratify=df["category"]
    )
    return (
        df.loc[idx_train, "clean_text"].values,
        df.loc[idx_test, "clean_text"].values,
        df.loc[idx_train, "category"].values,
        df.loc[idx_test, "category"].values,
        idx_train,
        idx_test,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["logreg", "mlp"], default="logreg",
                         help="logreg = linear baseline, mlp = small ANN (nonlinear)")
    parser.add_argument("--embedding", choices=["tfidf", "transformer"], default="tfidf")
    args = parser.parse_args()
    main(embedding_backend=args.embedding, model_type=args.model)
