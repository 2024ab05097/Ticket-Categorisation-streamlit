"""
orchestrator.py
-----------------
This is the "Agentic Orchestrator" node in the architecture diagram:

    Ticket Inflow -> Data Cleaning -> Feature Engineering -> Orchestrator Agent
    -> Stage-1 Request Classification -> Stage-2 Team Assignment
    -> Explainability Engine -> Confidence Filter -> Human Review / Automated Routing
    -> Feedback Loop -> Active Learning -> REST API Output

It's intentionally implemented as a plain Python class rather than a heavy
agent framework for the mid-sem milestone -- "agentic" here means each stage
is a self-contained, swappable component with a clear contract.

"Open/Closed Model Selection" tiering:
    Tier 1 (always runs): sklearn heads (Stage-1/Stage-2/Priority) -- cheap,
        milliseconds, fully local.
    Tier 2 (only for low-confidence tickets): local LLM via Ollama
        (llm_fallback.py) -- still fully local/offline, no API cost, just
        slower, so it's gated behind a confidence check rather than run on
        every ticket.
    Tier 3 (only if Tier 2 is also unavailable/uncertain): human review,
        via the confidence engine's escalate_to_human action.

No tier calls a paid/hosted model. Tier 2 is entirely optional -- if Ollama
isn't running, the orchestrator just skips it and falls through to Tier 3,
so nothing breaks in environments without a local LLM installed.
"""

from src.preprocessing import clean_dataframe, mask_dangerous_features, normalize_text
from src.confidence_engine import decide, HUMAN_REVIEW_THRESHOLD
from src.explainability import LinearWeightExplainer

SLA_HOURS = {"P1-Critical": 2, "P2-High": 8, "P3-Medium": 24, "P4-Low": 72}

# Below this Stage-1/Stage-2 confidence, try the local LLM before giving up
# to a human. Deliberately looser than HUMAN_REVIEW_THRESHOLD used by the
# confidence engine's final routing decision -- this just decides whether
# it's worth *asking* the LLM at all.
LLM_TRIGGER_THRESHOLD = 0.6


class TicketTriageOrchestrator:
    def __init__(self, embedder, stage1_model, stage2_model, priority_model,
                 explainer=None, llm_fallback=None):
        self.embedder = embedder
        self.stage1_model = stage1_model
        self.stage2_model = stage2_model
        self.priority_model = priority_model
        self.explainer = explainer
        self.llm_fallback = llm_fallback   # OllamaTicketClassifier instance, or None

    def process_ticket(self, ticket_id: str, title: str, description: str):
        # 1. Data Cleaning (single-ticket path mirrors clean_dataframe's steps)
        title_clean = mask_dangerous_features(title)
        desc_clean = mask_dangerous_features(description)
        text = normalize_text(f"{title_clean}. {desc_clean}")

        # 2. Feature Engineering
        X = self.embedder.transform([text])

        # 3. Stage-1 Request Classification
        category, cat_conf = self.stage1_model.predict_with_confidence(X)
        category, cat_conf = category[0], float(cat_conf[0])

        # 4. Stage-2 Team Assignment (conditioned on Stage-1 output)
        resolver_group, res_conf = self.stage2_model.predict_with_confidence(X, [category])
        resolver_group, res_conf = resolver_group[0], float(res_conf[0])

        # 5. Priority head
        priority, pri_conf = self.priority_model.predict_with_confidence(X)
        priority, pri_conf = priority[0], float(pri_conf[0])

        # 5b. Tier-2: local LLM second opinion, only when the cheap models are unsure
        model_tier = "sklearn"
        llm_reasoning = None
        weakest_conf = min(cat_conf, res_conf)
        if self.llm_fallback is not None and weakest_conf < LLM_TRIGGER_THRESHOLD:
            llm_result = self.llm_fallback.classify(title_clean, desc_clean)
            if llm_result.get("category"):
                model_tier = "local_llm"
                llm_reasoning = llm_result.get("reasoning")
                # Simple agreement policy: if the LLM agrees with the sklearn
                # category, boost confidence (two independent models agreeing
                # is meaningful signal). If it disagrees, defer to whichever
                # is more confident -- but always keep both visible in the
                # audit trail via llm_reasoning / model_tier.
                llm_conf = float(llm_result.get("confidence") or 0.0)
                if llm_result["category"] == category:
                    cat_conf = max(cat_conf, min(0.95, cat_conf + 0.2))
                elif llm_conf > cat_conf:
                    category, cat_conf = llm_result["category"], llm_conf
                if llm_result.get("resolver_group") and llm_conf > res_conf:
                    resolver_group, res_conf = llm_result["resolver_group"], llm_conf
                if llm_result.get("priority"):
                    priority = llm_result["priority"]

        # 6. Explainability Engine
        explanation = None
        if self.explainer is not None:
            try:
                class_index = list(self.stage1_model.classes_).index(category)
                explanation = self.explainer.explain(X, class_index)
            except Exception:
                explanation = None
        if llm_reasoning:
            explanation = (explanation or []) + [{"term": "local_llm_reasoning", "contribution": llm_reasoning}]

        # 7. Confidence Filter -> Human Review / Automated Routing
        decision = decide(
            ticket_id=ticket_id,
            category=category, category_confidence=cat_conf,
            resolver_group=resolver_group, resolver_confidence=res_conf,
            priority=priority, priority_confidence=pri_conf,
            sla_hours=SLA_HOURS.get(priority, 24),
            explanation=explanation,
        )
        decision.model_tier = model_tier
        return decision

    def process_batch(self, df):
        """df needs columns: ticket_id, title, description"""
        results = []
        for _, row in df.iterrows():
            results.append(self.process_ticket(row["ticket_id"], row["title"], row["description"]))
        return results
