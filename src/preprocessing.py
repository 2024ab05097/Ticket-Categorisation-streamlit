"""
preprocessing.py
-----------------
Implements Section 4 (Scope of Work) items:
    - Data preprocessing and cleansing
    - Dangerous feature removal (PII / secrets masking)
    - Basic normalization ahead of transformer embedding

Keep this module dependency-light (stdlib + pandas + re) so it can run
inside any ITSM data-egress boundary without extra approvals.
"""

import re
import pandas as pd

# --- "Dangerous feature" patterns -------------------------------------------------
# Anything that could leak PII/secrets into logs, embeddings, or LLM prompts.
# Extend this list with organization-specific patterns during Phase 2.
_PATTERNS = {
    "email": re.compile(r"[\w\.-]+@[\w\.-]+\.\w+"),
    "ip_address": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "phone": re.compile(r"\b\d{10}\b|\b\+\d{1,3}[\s-]?\d{6,12}\b"),
    "credential_like": re.compile(r"(?i)\b(password|pwd|secret|api[_-]?key|token)\s*[:=]\s*\S+"),
    "employee_id": re.compile(r"\b[A-Z]{2,4}\d{4,8}\b"),
}

_MASK = "<{}_REDACTED>"


def mask_dangerous_features(text: str) -> str:
    """Redact PII / secrets before the text ever reaches an embedding model or LLM."""
    if not isinstance(text, str):
        return ""
    cleaned = text
    for label, pattern in _PATTERNS.items():
        cleaned = pattern.sub(_MASK.format(label.upper()), cleaned)
    return cleaned


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\s<>_]", " ", text)  # keep mask tokens like <EMAIL_REDACTED>
    return text.strip()


def clean_dataframe(df: pd.DataFrame, text_cols=("title", "description")) -> pd.DataFrame:
    """Full Phase-2 pipeline: dedupe -> handle missing -> mask -> normalize -> merge text."""
    df = df.copy()

    # 1. Handle missing values
    for col in text_cols:
        df[col] = df[col].fillna("")

    # 2. Deduplicate near-identical tickets (exact match on title+description here;
    #    swap for MinHash/embedding-based dedupe once volumes get large)
    df["_dedupe_key"] = (df["title"].str.strip().str.lower() + "||" +
                          df["description"].str.strip().str.lower())
    before = len(df)
    df = df.drop_duplicates(subset="_dedupe_key").drop(columns="_dedupe_key")
    removed = before - len(df)

    # 3. Dangerous feature removal
    for col in text_cols:
        df[col] = df[col].apply(mask_dangerous_features)

    # 4. Normalize + build the single text field the models consume
    df["clean_text"] = (df["title"] + ". " + df["description"]).apply(normalize_text)

    df.attrs["duplicates_removed"] = removed
    return df


if __name__ == "__main__":
    sample = pd.DataFrame({
        "title": ["VPN not connecting", "VPN not connecting"],
        "description": [
            "Contact me at john.doe@company.com or 9876543210, password: hunter2",
            "Contact me at john.doe@company.com or 9876543210, password: hunter2",
        ],
    })
    out = clean_dataframe(sample)
    print(out[["clean_text"]])
    print("duplicates_removed:", out.attrs["duplicates_removed"])
