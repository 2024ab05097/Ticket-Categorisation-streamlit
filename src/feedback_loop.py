"""
feedback_loop.py
------------------
Implements Objectives: "Build feedback loop using active learning",
"Build human-in-the-loop validation workflows", "Enable continuous
learning through RLHF/DPO inspired feedback loops".

This is the "Feedback Loop -> Active Learning" node in the architecture
diagram. Two responsibilities:

    1. FeedbackStore: durable log of every human correction (reviewer
       accepted or overrode a model prediction). Append-only JSONL so it's
       trivially diffable/auditable and needs zero infra (swap for a real
       DB table later -- the interface won't change).

    2. select_for_active_learning(): given a batch of live routing
       decisions, rank the lowest-confidence ones so a human reviewer
       spends their time where it matters most, rather than reviewing
       randomly. This is the "Active Learning" box in the diagram.

The (model_prediction, human_label) pairs logged here are also exactly the
shape you need for a DPO-style (rejected, chosen) preference dataset later,
per the strategy doc -- we don't build the RLHF/DPO trainer itself for the
mid-sem milestone, but the data format is ready for it.
"""

import json
import os
import threading
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Optional, List, Dict, Iterable

DEFAULT_STORE_PATH = "artifacts/feedback_log.jsonl"


@dataclass
class FeedbackRecord:
    ticket_id: str
    field: str                       # "category" | "resolver_group" | "priority"
    model_prediction: str
    model_confidence: float
    human_label: str
    reviewer_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def is_disagreement(self) -> bool:
        return self.model_prediction != self.human_label


class FeedbackStore:
    """Append-only JSONL feedback log. Thread-safe for single-process use
    (FastAPI's default sync workers); swap for a real DB if you move to
    multi-process serving (see IMPLEMENTATION_STRATEGY.md Phase 6)."""

    def __init__(self, path: str = DEFAULT_STORE_PATH):
        self.path = path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        if not os.path.exists(self.path):
            open(self.path, "a").close()

    def log(self, record: FeedbackRecord) -> None:
        with self._lock:
            with open(self.path, "a") as f:
                f.write(json.dumps(asdict(record)) + "\n")

    def load_all(self) -> List[FeedbackRecord]:
        records = []
        if not os.path.exists(self.path):
            return records
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(FeedbackRecord(**json.loads(line)))
        return records

    def disagreement_rate(self, field_name: Optional[str] = None) -> float:
        """Fraction of logged feedback where the human overrode the model.
        Track this over successive retraining cycles as the 'demonstration
        of continuous performance improvement' evidence (Section 9)."""
        records = self.load_all()
        if field_name is not None:
            records = [r for r in records if r.field == field_name]
        if not records:
            return 0.0
        return sum(r.is_disagreement for r in records) / len(records)

    def corrected_training_rows(self, field_name: str) -> List[Dict]:
        """Human-corrected (text-free) label pairs for a given field, ready
        to be merged back into a retraining set by ticket_id lookup."""
        return [
            {"ticket_id": r.ticket_id, "label": r.human_label}
            for r in self.load_all()
            if r.field == field_name and r.is_disagreement
        ]


def select_for_active_learning(
    decisions: Iterable[dict],
    n: int = 20,
    confidence_key: str = "category_confidence",
) -> List[dict]:
    """Given a batch of routing decisions (dict form of RoutingDecision, or
    any dict with a confidence field), return the `n` lowest-confidence ones
    -- these are the tickets where a human reviewer's time is best spent,
    rather than reviewing a random sample."""
    ranked = sorted(decisions, key=lambda d: d.get(confidence_key, 1.0))
    return ranked[:n]
