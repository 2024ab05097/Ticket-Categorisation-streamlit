"""
api.py
-------
Implements Objectives: "Develop RESTful API for real-time analysis",
"Enable real-time NLP-based triaging", "Integration with enterprise ITSM
platforms".

Run locally:
    pip install fastapi uvicorn
    uvicorn src.api:app --reload --port 8000

Endpoints:
    POST /triage            -> classify + route a single ticket
    POST /triage/batch      -> classify + route a list of tickets
    POST /feedback          -> log a human correction (closes the active-learning loop)
    GET  /health            -> liveness check for ITSM integration monitoring
"""

from typing import List, Optional
import joblib

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except ImportError as e:
    raise ImportError(
        "FastAPI not installed. Run `pip install fastapi uvicorn pydantic` "
        "to serve the API. The rest of the pipeline (train.py, evaluate.py) "
        "works without it."
    ) from e

from src.orchestrator import TicketTriageOrchestrator
from src.explainability import LinearWeightExplainer
from src.feedback_loop import FeedbackStore, FeedbackRecord
from src.llm_fallback import get_llm_fallback

ARTIFACT_DIR = "artifacts"

app = FastAPI(title="Ticket Categorisation & Triaging API", version="0.1.0")

_state = {}


class TicketIn(BaseModel):
    ticket_id: str
    title: str
    description: str


class BatchIn(BaseModel):
    tickets: List[TicketIn]


class FeedbackIn(BaseModel):
    ticket_id: str
    field: str                # "category" | "resolver_group" | "priority"
    model_prediction: str
    model_confidence: float
    human_label: str
    reviewer_id: Optional[str] = None


@app.on_event("startup")
def load_artifacts():
    embedder = joblib.load(f"{ARTIFACT_DIR}/embedder.joblib")
    stage1 = joblib.load(f"{ARTIFACT_DIR}/stage1_category.joblib")
    stage2 = joblib.load(f"{ARTIFACT_DIR}/stage2_resolver.joblib")
    priority = joblib.load(f"{ARTIFACT_DIR}/priority.joblib")
    explainer = None
    try:
        explainer = LinearWeightExplainer(stage1.model, embedder.get_feature_names())
    except Exception:
        pass
    _state["orchestrator"] = TicketTriageOrchestrator(
        embedder, stage1, stage2, priority, explainer,
        llm_fallback=get_llm_fallback(),   # None if Ollama isn't running -- API still works
    )
    _state["feedback_store"] = FeedbackStore()


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "orchestrator" in _state}


@app.post("/triage")
def triage(ticket: TicketIn):
    if "orchestrator" not in _state:
        raise HTTPException(503, "Model artifacts not loaded yet.")
    decision = _state["orchestrator"].process_ticket(ticket.ticket_id, ticket.title, ticket.description)
    return decision.__dict__


@app.post("/triage/batch")
def triage_batch(batch: BatchIn):
    if "orchestrator" not in _state:
        raise HTTPException(503, "Model artifacts not loaded yet.")
    out = []
    for t in batch.tickets:
        d = _state["orchestrator"].process_ticket(t.ticket_id, t.title, t.description)
        out.append(d.__dict__)
    return {"results": out}


@app.post("/feedback")
def feedback(fb: FeedbackIn):
    record = FeedbackRecord(
        ticket_id=fb.ticket_id,
        field=fb.field,
        model_prediction=fb.model_prediction,
        model_confidence=fb.model_confidence,
        human_label=fb.human_label,
        reviewer_id=fb.reviewer_id,
    )
    _state["feedback_store"].log(record)
    return {"status": "logged"}
