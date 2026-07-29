"""
llm_fallback.py
-----------------
Implements the "Open/Closed Model Selection" node in the architecture
diagram, restricted to fully local/offline models per your constraint:
no hosted API, no per-token cost.

How it works:
    - Uses Ollama (https://ollama.com) as the local model runtime. Ollama
      downloads open-weight model files ONCE (e.g. `ollama pull mistral`)
      and serves them from a local HTTP endpoint (localhost:11434) with no
      further internet access needed. There is no per-request billing --
      you're just spending your own CPU/GPU cycles.
    - This module is only invoked as a FALLBACK, when the cheap sklearn
      classifiers (Stage-1/Stage-2/Priority) are not confident. Most
      tickets never reach this code path, which is exactly why it doesn't
      matter that an LLM call is slower/heavier than a TF-IDF+LogReg call.
    - Any Ollama-served open-weight model works: "mistral", "llama3",
      "phi3", "qwen2.5", etc. Pick based on what fits your hardware.

Setup (one-time, then fully offline):
    1. Install Ollama: https://ollama.com/download
    2. Pull a model:   `ollama pull mistral`
    3. Ollama runs a local server automatically on localhost:11434

No API key. No cloud call. No token billing.
"""

import json
import urllib.request
import urllib.error

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "mistral"

_PROMPT_TEMPLATE = """You are an IT service desk triage assistant. Read the ticket and respond
with ONLY a JSON object (no other text) in this exact shape:

{{"category": "<one of: Network, Hardware, Software, Access Management, Cloud Infra>",
  "resolver_group": "<best-guess resolver team name, short>",
  "priority": "<one of: P1-Critical, P2-High, P3-Medium, P4-Low>",
  "confidence": <float 0-1, your own certainty>,
  "reasoning": "<one sentence explaining the classification>"}}

Ticket title: {title}
Ticket description: {description}
"""


class OllamaUnavailableError(RuntimeError):
    pass


class OllamaTicketClassifier:
    """Thin wrapper around a local Ollama server. No external dependency
    beyond the Python standard library, so it doesn't bloat requirements.txt
    for people who don't want the LLM fallback at all."""

    def __init__(self, model: str = DEFAULT_MODEL, url: str = OLLAMA_URL, timeout: int = 30):
        self.model = model
        self.url = url
        self.timeout = timeout

    def classify(self, title: str, description: str) -> dict:
        prompt = _PROMPT_TEMPLATE.format(title=title, description=description)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",       # ask Ollama to constrain output to valid JSON
            "options": {"temperature": 0.1},
        }
        req = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            raise OllamaUnavailableError(
                f"Could not reach local Ollama server at {self.url}. "
                f"Is Ollama running? (`ollama serve`, and `ollama pull {self.model}` "
                f"if you haven't already). Original error: {e}"
            ) from e

        raw_text = body.get("response", "{}")
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            # Model didn't return clean JSON despite format="json" -- degrade gracefully
            # rather than crash the pipeline.
            parsed = {"category": None, "resolver_group": None, "priority": None,
                       "confidence": 0.0, "reasoning": "LLM returned unparsable output."}
        return parsed


def get_llm_fallback(model: str = DEFAULT_MODEL):
    """Factory used by orchestrator.py. Returns None (and lets the orchestrator
    skip the LLM tier entirely) if Ollama isn't reachable, instead of crashing
    the whole ticket pipeline over an optional component."""
    classifier = OllamaTicketClassifier(model=model, timeout=3)  # short timeout for healthcheck only
    try:
        classifier.classify("healthcheck", "healthcheck")
        classifier.timeout = 30  # restore normal timeout for real classification calls
        return classifier
    except OllamaUnavailableError:
        return None
