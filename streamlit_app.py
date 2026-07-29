"""
streamlit_app.py
------------------
Interactive UI for the ticket triage pipeline. Wraps the exact same
components used by run_pipeline.py and src/api.py -- no separate logic path,
so what you see here is what the REST API would return too.

Run:
    pip install -r requirements.txt
    streamlit run streamlit_app.py

Pages (sidebar):
    1. Single Ticket Triage    -- type a ticket, see the full routing decision
                                  + term-level explanation + log a correction
    2. Batch Routing            -- upload a CSV or generate a synthetic batch,
                                  route all of it, see governance charts,
                                  download results
    3. Feedback & Active Learning -- feedback log, disagreement rate,
                                  lowest-confidence tickets to review next
    4. Governance Dashboard     -- live-adjustable AUTO_ROUTE / HUMAN_REVIEW
                                  thresholds, accuracy vs. ground truth (for
                                  synthetic batches)

If no trained models exist yet in artifacts/, the sidebar offers a
"Train models now" button that calls src/train.py directly.
"""

import os
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
import joblib

from src.data_generator import generate_dataset
from src.explainability import LinearWeightExplainer
from src.orchestrator import TicketTriageOrchestrator
from src.feedback_loop import FeedbackStore, FeedbackRecord, select_for_active_learning
from src.llm_fallback import get_llm_fallback
import src.confidence_engine as confidence_engine
try:
    from src import train as train_module
except Exception:
    train_module = None
    
ARTIFACT_DIR = "artifacts"
FEEDBACK_PATH = f"{ARTIFACT_DIR}/feedback_log.jsonl"

ACTION_COLORS = {
    "auto_route": "#2ecc71",
    "route_with_review": "#f39c12",
    "escalate_to_human": "#e74c3c",
}
ACTION_LABELS = {
    "auto_route": "✅ Auto-routed",
    "route_with_review": "🟡 Routed (needs review)",
    "escalate_to_human": "🔴 Escalated to human",
}

st.set_page_config(page_title="Ticket Triage Console", page_icon="🎫", layout="wide")


# --------------------------------------------------------------------------
# Artifact / orchestrator loading
# --------------------------------------------------------------------------

def artifacts_exist() -> bool:
    required = ["embedder.joblib", "stage1_category.joblib", "stage2_resolver.joblib", "priority.joblib"]
    return all(os.path.exists(f"{ARTIFACT_DIR}/{f}") for f in required)


@st.cache_resource(show_spinner=False)
def load_orchestrator(_artifact_version: int):
    """_artifact_version is unused inside the function but forces Streamlit
    to rebuild (rather than reuse) the cached orchestrator after a retrain,
    since cache_resource keys on the arguments passed in."""
    embedder = joblib.load(f"{ARTIFACT_DIR}/embedder.joblib")
    stage1 = joblib.load(f"{ARTIFACT_DIR}/stage1_category.joblib")
    stage2 = joblib.load(f"{ARTIFACT_DIR}/stage2_resolver.joblib")
    priority = joblib.load(f"{ARTIFACT_DIR}/priority.joblib")

    explainer = None
    try:
        explainer = LinearWeightExplainer(stage1.model, embedder.get_feature_names())
    except Exception:
        explainer = None

    llm_fallback = get_llm_fallback()
    orchestrator = TicketTriageOrchestrator(
        embedder, stage1, stage2, priority,
        explainer=explainer, llm_fallback=llm_fallback,
    )
    return orchestrator, (llm_fallback is not None)


@st.cache_resource(show_spinner=False)
def get_feedback_store():
    return FeedbackStore(path=FEEDBACK_PATH)


def run_training(model_type: str, n_train: int):
    if train_module is None:
        st.error("Training module unavailable in Streamlit deployment.")
        return

    data_csv = "data/synthetic_tickets.csv"
    if os.path.exists(data_csv):
        os.remove(data_csv)

    train_module.main(
        embedding_backend="tfidf",
        model_type=model_type
    )

# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------

st.session_state.setdefault("artifact_version", 0)
st.session_state.setdefault("batch_results", None)
st.session_state.setdefault("batch_truth", None)
st.session_state.setdefault("last_decision", None)
st.session_state.setdefault("last_ticket", None)


# --------------------------------------------------------------------------
# Sidebar: model status / training / thresholds / navigation
# --------------------------------------------------------------------------

st.sidebar.title("🎫 Ticket Triage Console")
st.sidebar.caption("Agentic ITSM ticket categorisation & triaging")

if not artifacts_exist():
    st.sidebar.warning("No trained models found in `artifacts/` yet.")
    with st.sidebar.form("train_form_initial"):
        model_type = st.selectbox("Model type", ["logreg", "mlp"], help="logreg = linear, fast, explainable. mlp = small ANN.")
        n_train = st.number_input("Synthetic training tickets", 200, 5000, 1500, step=100)
        submitted = st.form_submit_button("Train models now", type="primary")
    if submitted:
        with st.spinner("Training Stage-1 / Stage-2 / Priority models..."):
            run_training(model_type, n_train)
        st.session_state.artifact_version += 1
        st.cache_resource.clear()
        st.rerun()
    st.title("Ticket Triage Console")
    st.info("⬅️ Train the models from the sidebar to get started. This trains on synthetic ITSM tickets in a few seconds.")
    st.stop()

st.sidebar.success("Models loaded ✅")
with st.sidebar.expander("Retrain models"):
    model_type_r = st.selectbox("Model type", ["logreg", "mlp"], key="retrain_model_type")
    n_train_r = st.number_input("Synthetic training tickets", 200, 5000, 1500, step=100, key="retrain_n")
    if st.button("Retrain now"):
        with st.spinner("Retraining..."):
            run_training(model_type_r, n_train_r)
        st.session_state.artifact_version += 1
        st.cache_resource.clear()
        st.rerun()

orchestrator, llm_available = load_orchestrator(st.session_state.artifact_version)
feedback_store = get_feedback_store()

st.sidebar.caption(f"Local LLM fallback (Ollama): {'🟢 available' if llm_available else '⚪ not running'}")

st.sidebar.divider()
st.sidebar.subheader("Governance thresholds")
st.sidebar.caption("Changes apply immediately to every ticket routed below.")
auto_th = st.sidebar.slider("Auto-route threshold", 0.50, 0.99, float(confidence_engine.AUTO_ROUTE_THRESHOLD), 0.01)
review_th = st.sidebar.slider("Human-review threshold", 0.10, auto_th, min(float(confidence_engine.HUMAN_REVIEW_THRESHOLD), auto_th), 0.01)
confidence_engine.AUTO_ROUTE_THRESHOLD = auto_th
confidence_engine.HUMAN_REVIEW_THRESHOLD = review_th

st.sidebar.divider()
page = st.sidebar.radio(
    "Navigate",
    ["Single Ticket Triage", "Batch Routing", "Feedback & Active Learning", "Governance Dashboard"],
)


# --------------------------------------------------------------------------
# Page: Single Ticket Triage
# --------------------------------------------------------------------------

if page == "Single Ticket Triage":
    st.title("Single Ticket Triage")
    st.caption("Cleaning → embedding → Stage-1/Stage-2/Priority → explainability → confidence routing, live.")

    with st.form("ticket_form"):
        ticket_id = st.text_input("Ticket ID", value=f"TCK-{np.random.randint(100000, 999999)}")
        title = st.text_input("Title", placeholder="e.g. VPN not connecting")
        description = st.text_area(
            "Description", height=120,
            placeholder="e.g. Critical, all users affected, production down. Contact me at john@company.com",
        )
        submitted = st.form_submit_button("Triage ticket", type="primary")

    if submitted:
        if not title.strip() and not description.strip():
            st.warning("Enter a title or description first.")
        else:
            decision = orchestrator.process_ticket(ticket_id, title, description)
            st.session_state.last_decision = decision
            st.session_state.last_ticket = {"ticket_id": ticket_id, "title": title, "description": description}

    d = st.session_state.last_decision
    if d is not None:
        st.markdown(f"### {ACTION_LABELS.get(d.action, d.action)}")
        st.caption(d.reason)

        m1, m2, m3 = st.columns(3)
        m1.metric("Category", d.category, f"{d.category_confidence:.0%} confidence")
        m2.metric("Resolver group", d.resolver_group, f"{d.resolver_confidence:.0%} confidence")
        m3.metric("Priority", d.priority, f"SLA {d.sla_hours}h")

        if d.misrouting_flag:
            st.error("⚠️ Misrouting flag raised — resolver-group confidence is low/unstable for this category.")

        if d.model_tier == "local_llm":
            st.caption("🤖 This decision was informed by the local LLM fallback tier (sklearn heads were unsure).")

        if d.explanation:
            term_rows = [e for e in d.explanation if isinstance(e.get("contribution"), (int, float))]
            if term_rows:
                st.subheader("Why this classification? (top contributing terms)")
                exp_df = pd.DataFrame(term_rows)
                chart = (
                    alt.Chart(exp_df)
                    .mark_bar()
                    .encode(
                        x=alt.X("contribution:Q"),
                        y=alt.Y("term:N", sort="-x", title=None),
                        color=alt.condition(alt.datum.contribution > 0, alt.value("#2ecc71"), alt.value("#e74c3c")),
                        tooltip=["term", "contribution"],
                    )
                    .properties(height=28 * len(term_rows) + 40)
                )
                st.altair_chart(chart, use_container_width=True)
            llm_notes = [e for e in d.explanation if e.get("term") == "local_llm_reasoning"]
            for note in llm_notes:
                st.info(f"🤖 Local LLM reasoning: {note['contribution']}")
        else:
            st.caption("No term-level explanation available (linear explainer needs a `logreg` Stage-1 head).")

        st.divider()
        st.subheader("Human review")
        with st.form("feedback_form_single"):
            correct = st.radio("Was the resolver-group assignment correct?", ["Yes", "No"], horizontal=True)
            corrected_group = st.text_input("Correct resolver group", disabled=(correct == "Yes"))
            reviewer = st.text_input("Reviewer name/ID", value="demo_reviewer")
            log_it = st.form_submit_button("Log feedback")
        if log_it:
            human_label = d.resolver_group if correct == "Yes" else (corrected_group.strip() or d.resolver_group)
            feedback_store.log(FeedbackRecord(
                ticket_id=d.ticket_id, field="resolver_group",
                model_prediction=d.resolver_group, model_confidence=d.resolver_confidence,
                human_label=human_label, reviewer_id=reviewer,
            ))
            st.success("Feedback logged to the active-learning store.")


# --------------------------------------------------------------------------
# Page: Batch Routing
# --------------------------------------------------------------------------

elif page == "Batch Routing":
    st.title("Batch Routing")
    st.caption("Upload a CSV (ticket_id, title, description) or generate a synthetic batch to see routing at scale.")

    tab_upload, tab_synth = st.tabs(["Upload CSV", "Generate synthetic batch"])
    df_in = None

    with tab_upload:
        uploaded = st.file_uploader("CSV with columns: ticket_id, title, description", type="csv")
        if uploaded is not None:
            df_in = pd.read_csv(uploaded)
            st.session_state.batch_truth = None  # no ground truth for uploaded data

    with tab_synth:
        n = st.slider("Number of synthetic tickets", 10, 1000, 100, step=10)
        if st.button("Generate & route", type="primary"):
            df_in = generate_dataset(n)
            st.session_state.batch_truth = df_in.reset_index(drop=True)

    if df_in is not None:
        missing = [c for c in ["ticket_id", "title", "description"] if c not in df_in.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            with st.spinner(f"Routing {len(df_in)} tickets..."):
                decisions = orchestrator.process_batch(df_in)
            st.session_state.batch_results = pd.DataFrame([dec.__dict__ for dec in decisions])
            if st.session_state.batch_truth is None:
                st.session_state.batch_truth = df_in.reset_index(drop=True)

    results = st.session_state.batch_results
    if results is not None:
        st.subheader(f"Routed {len(results)} tickets")

        c1, c2, c3 = st.columns(3)
        c1.metric("Auto-routed", f"{(results['action'] == 'auto_route').mean():.0%}")
        c2.metric("Routed w/ review", f"{(results['action'] == 'route_with_review').mean():.0%}")
        c3.metric("Escalated", f"{(results['action'] == 'escalate_to_human').mean():.0%}")

        action_counts = results["action"].value_counts().rename_axis("action").reset_index(name="count")
        chart = (
            alt.Chart(action_counts)
            .mark_bar()
            .encode(
                x=alt.X("action:N", title=None),
                y=alt.Y("count:Q"),
                color=alt.Color(
                    "action:N",
                    scale=alt.Scale(domain=list(ACTION_COLORS.keys()), range=list(ACTION_COLORS.values())),
                    legend=None,
                ),
                tooltip=["action", "count"],
            )
        )
        st.altair_chart(chart, use_container_width=True)

        display_cols = ["ticket_id", "category", "category_confidence", "resolver_group",
                         "resolver_confidence", "priority", "action", "misrouting_flag", "reason"]

        def _row_style(row):
            color = ACTION_COLORS.get(row["action"], "#ffffff")
            return [f"background-color: {color}22"] * len(row)

        st.dataframe(
            results[display_cols].style.apply(_row_style, axis=1),
            use_container_width=True, height=420,
        )

        csv_bytes = results.to_csv(index=False).encode("utf-8")
        st.download_button("Download results as CSV", csv_bytes, "triage_results.csv", "text/csv")


# --------------------------------------------------------------------------
# Page: Feedback & Active Learning
# --------------------------------------------------------------------------

elif page == "Feedback & Active Learning":
    st.title("Feedback & Active Learning")

    all_fb = feedback_store.load_all()
    c1, c2 = st.columns(2)
    c1.metric("Total feedback records", len(all_fb))
    if all_fb:
        c2.metric("Overall disagreement rate", f"{feedback_store.disagreement_rate():.1%}")
        fb_df = pd.DataFrame([r.__dict__ for r in all_fb])
        st.dataframe(fb_df, use_container_width=True)
        csv_bytes = fb_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download feedback log", csv_bytes, "feedback_log.csv", "text/csv")
    else:
        st.info("No feedback logged yet. Log some from the 'Single Ticket Triage' page.")

    st.divider()
    st.subheader("Active learning queue")
    st.caption("Tickets ranked lowest-confidence-first from the most recent batch, so a reviewer's time goes where it matters most.")

    if st.session_state.batch_results is not None:
        conf_field = st.selectbox(
            "Rank by", ["resolver_confidence", "category_confidence", "priority_confidence"],
        )
        n_review = st.slider("How many tickets to review next", 1, 30, 5)
        candidates = st.session_state.batch_results.to_dict("records")
        picked = select_for_active_learning(candidates, n=n_review, confidence_key=conf_field)
        picked_df = pd.DataFrame(picked)[["ticket_id", "category", "resolver_group", "priority", "action", conf_field]]
        st.dataframe(picked_df, use_container_width=True)
    else:
        st.info("Run a batch on the 'Batch Routing' page first to populate the active-learning queue.")


# --------------------------------------------------------------------------
# Page: Governance Dashboard
# --------------------------------------------------------------------------

else:
    st.title("Governance Dashboard")
    st.write(
        f"**Auto-route threshold:** {confidence_engine.AUTO_ROUTE_THRESHOLD:.2f}  |  "
        f"**Human-review threshold:** {confidence_engine.HUMAN_REVIEW_THRESHOLD:.2f}"
    )
    st.caption("Adjust these from the sidebar — they apply immediately, no retraining needed.")

    results = st.session_state.batch_results
    truth = st.session_state.batch_truth

    if results is not None:
        st.subheader("Latest batch — governance metrics")
        st.write(f"Misrouting flag rate: **{results['misrouting_flag'].mean():.1%}**")

        if truth is not None and "priority" in truth.columns:
            p1p2 = results[truth["priority"].isin(["P1-Critical", "P2-High"])]
            if len(p1p2):
                pct = (p1p2["action"] != "auto_route").mean()
                st.write(f"P1/P2 tickets with a human in the loop: **{pct:.1%}** (target: 100%)")

        if truth is not None and {"category", "resolver_group", "priority"}.issubset(truth.columns):
            merged = results.merge(
                truth[["ticket_id", "category", "resolver_group", "priority"]],
                on="ticket_id", suffixes=("_pred", "_true"),
            )
            cat_acc = (merged["category_pred"] == merged["category_true"]).mean()
            res_acc = (merged["resolver_group_pred"] == merged["resolver_group_true"]).mean()
            pri_acc = (merged["priority_pred"] == merged["priority_true"]).mean()

            st.subheader("Accuracy vs. ground truth (synthetic batch only)")
            c1, c2, c3 = st.columns(3)
            c1.metric("Category accuracy", f"{cat_acc:.1%}")
            c2.metric("Resolver-group accuracy", f"{res_acc:.1%}")
            c3.metric("Priority accuracy", f"{pri_acc:.1%}")
        else:
            st.caption("Upload a CSV without ground-truth labels? Accuracy metrics need a synthetic batch (Batch Routing → Generate synthetic batch).")
    else:
        st.info("Run a synthetic batch on 'Batch Routing' to see live governance metrics here.")
