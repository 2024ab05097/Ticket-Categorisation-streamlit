"""
confidence_engine.py
----------------------
Implements Objectives: "Implement confidence-aware routing", "Support
SLA-aware triaging", "Detect and flag misrouting cases".

This is the "Confidence Filter" + "Human Review / Automated Routing" node
in the architecture diagram. It is a deliberately simple, auditable rules
layer on top of the ML confidence scores -- keep the ML opaque parts small
and the governance/routing logic transparent, which is the core pitch of
"Operational Trustworthiness" in the Design Considerations section.
"""

from dataclasses import dataclass, field
from typing import Optional


# Tune these during Phase 7 (Testing & Evaluation) against a labeled
# validation set; start conservative and loosen as trust in the model grows.
AUTO_ROUTE_THRESHOLD = 0.85     # >= this -> fully automated routing
HUMAN_REVIEW_THRESHOLD = 0.55   # between this and AUTO -> route but flag for async review
# < HUMAN_REVIEW_THRESHOLD -> hard escalate to human triage queue before routing


@dataclass
class RoutingDecision:
    ticket_id: str
    category: str
    category_confidence: float
    resolver_group: str
    resolver_confidence: float
    priority: str
    priority_confidence: float
    sla_hours: int
    action: str                       # "auto_route" | "route_with_review" | "escalate_to_human"
    misrouting_flag: bool
    explanation: Optional[list] = field(default=None)
    reason: str = ""
    model_tier: str = "sklearn"


def _overall_confidence(cat_conf, res_conf, pri_conf) -> float:
    # Weighted toward resolver-group confidence since a wrong team assignment
    # is the costliest failure mode (rework + SLA breach risk).
    return 0.3 * cat_conf + 0.5 * res_conf + 0.2 * pri_conf


def decide(
    ticket_id: str,
    category: str, category_confidence: float,
    resolver_group: str, resolver_confidence: float,
    priority: str, priority_confidence: float,
    sla_hours: int,
    explanation=None,
) -> RoutingDecision:
    overall = _overall_confidence(category_confidence, resolver_confidence, priority_confidence)

    misrouting_flag = (
        resolver_group == "UNKNOWN_RESOLVER_GROUP"
        or resolver_confidence < 0.4
        or (category_confidence > 0.8 and resolver_confidence < 0.3)  # confident category, confused team -> smells like Gap-3 (overlapping resolver groups)
    )

    if misrouting_flag:
        action = "escalate_to_human"
        reason = "Potential misrouting detected (low/unstable resolver-group confidence)."
    elif overall >= AUTO_ROUTE_THRESHOLD and priority != "P1-Critical":
        action = "auto_route"
        reason = f"Overall confidence {overall:.2f} >= auto-route threshold."
    elif priority == "P1-Critical":
        # Always keep a human in the loop for critical-priority tickets regardless
        # of model confidence -- this is a governance/trust requirement, not an
        # accuracy one (see Design Considerations: Operational Trustworthiness).
        action = "route_with_review"
        reason = "P1-Critical tickets always get human co-sign before routing."
    elif overall >= HUMAN_REVIEW_THRESHOLD:
        action = "route_with_review"
        reason = f"Overall confidence {overall:.2f} in the review band."
    else:
        action = "escalate_to_human"
        reason = f"Overall confidence {overall:.2f} below human-review threshold."

    return RoutingDecision(
        ticket_id=ticket_id,
        category=category, category_confidence=round(float(category_confidence), 3),
        resolver_group=resolver_group, resolver_confidence=round(float(resolver_confidence), 3),
        priority=priority, priority_confidence=round(float(priority_confidence), 3),
        sla_hours=sla_hours,
        action=action,
        misrouting_flag=misrouting_flag,
        explanation=explanation,
        reason=reason,
    )
