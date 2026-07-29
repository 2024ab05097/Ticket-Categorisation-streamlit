"""
data_generator.py
------------------
Generates a synthetic ITSM ticket dataset that mimics the structure described
in the dissertation synopsis (ticket title, description, category, priority,
resolver group, SLA hours). Use this to smoke-test the pipeline before you
plug in the real Infosys ITSM export. Swap `load_real_dataset()` in
`train.py` for this once real data is available -- nothing else in the
pipeline needs to change because everything downstream consumes the same
DataFrame schema.
"""

import random
import pandas as pd

random.seed(42)

CATEGORIES = {
    "Network": ["VPN not connecting", "WiFi drops intermittently", "Cannot reach internal server",
                "DNS resolution failing", "Slow network on floor 3"],
    "Hardware": ["Laptop won't power on", "Monitor flickering", "Keyboard keys unresponsive",
                 "Printer not detected", "Battery draining fast"],
    "Software": ["Application crashes on launch", "License activation error", "Update stuck at 40%",
                 "Outlook not syncing", "Excel macros disabled unexpectedly"],
    "Access Management": ["Need access to shared drive", "Password reset required",
                           "MFA device lost", "Account locked after failed logins",
                           "Request elevated admin rights"],
    "Cloud Infra": ["EC2 instance unreachable", "S3 bucket permission denied", "Kubernetes pod crashlooping",
                     "Azure VM high CPU alert", "Storage quota exceeded"],
}

RESOLVER_GROUPS = {
    "Network": ["NetOps-L1", "NetOps-L2"],
    "Hardware": ["Field-Support", "Hardware-Depot"],
    "Software": ["App-Support-L1", "App-Support-L2"],
    "Access Management": ["IAM-Team", "ServiceDesk-L1"],
    "Cloud Infra": ["CloudOps-L1", "CloudOps-L2"],
}

PRIORITIES = ["P1-Critical", "P2-High", "P3-Medium", "P4-Low"]
PRIORITY_WEIGHTS = [0.05, 0.20, 0.45, 0.30]  # realistic class imbalance, matches Gap-5 in synopsis

SLA_HOURS = {"P1-Critical": 2, "P2-High": 8, "P3-Medium": 24, "P4-Low": 72}


_USERS = ["Priya", "Arjun", "Rahul", "Divya", "Karthik", "Sneha", "Vikram", "Ananya", "Rohit", "Meera"]
_LOCATIONS = ["floor 2", "Chennai office", "remote setup", "the Bangalore branch", "the datacenter", "the client site"]


_URGENCY_PHRASES = {
    "P1-Critical": ["This is a production outage, all users affected, need immediate help.",
                    "Critical system down, business operations halted."],
    "P2-High": ["This is urgent and blocking my work, please expedite.",
                "High impact issue affecting the whole team."],
    "P3-Medium": ["This is a moderate issue, please address when possible.",
                  "Not blocking work but should be fixed soon."],
    "P4-Low": ["This is a minor issue, no rush.",
               "Low priority, just flagging for awareness."],
}


def _make_description(category: str, priority: str) -> str:
    base = random.choice(CATEGORIES[category])
    fillers = [
        f"This started happening around {random.randint(1,11)} AM today.",
        f"Impacting {random.choice(_USERS)}'s team at {random.choice(_LOCATIONS)}.",
        "Tried restarting but issue persists.",
        "Happens intermittently, hard to reproduce.",
        f"Reported by {random.choice(_USERS)}, ticket raised from {random.choice(_LOCATIONS)}.",
        "Second time this has occurred this week.",
    ]
    chosen = random.sample(fillers, k=random.randint(1, 2))
    urgency = random.choice(_URGENCY_PHRASES[priority])
    return f"{base}. {' '.join(chosen)} {urgency}".strip()


def generate_dataset(n: int = 3000) -> pd.DataFrame:
    """Generate a synthetic labeled ITSM ticket dataset.

    Returns a DataFrame with columns:
        ticket_id, title, description, category, priority, resolver_group, sla_hours
    """
    rows = []
    categories = list(CATEGORIES.keys())
    for i in range(n):
        category = random.choice(categories)
        title = random.choice(CATEGORIES[category])
        priority = random.choices(PRIORITIES, weights=PRIORITY_WEIGHTS, k=1)[0]
        description = _make_description(category, priority)

        # Give resolver-group assignment a learnable signal (L1 vs L2) tied to
        # complexity cues in the text / priority, rather than pure noise --
        # this mirrors real ITSM practice where recurring/high-priority issues
        # escalate straight to L2 specialists.
        l1_group, l2_group = RESOLVER_GROUPS[category]
        is_recurring_or_multiuser = ("second time" in description.lower()
                                      or "team at" in description.lower())
        if priority in ("P1-Critical", "P2-High") or is_recurring_or_multiuser:
            resolver_group = l2_group if random.random() < 0.8 else l1_group
        else:
            resolver_group = l1_group if random.random() < 0.85 else l2_group

        # inject a small amount of label noise / misrouting to simulate Gap-3
        # ("overlapping resolver groups cause confusion") from the literature review
        if random.random() < 0.03:
            resolver_group = random.choice(sum(RESOLVER_GROUPS.values(), []))
        rows.append({
            "ticket_id": f"TCK-{100000 + i}",
            "title": title,
            "description": description,
            "category": category,
            "priority": priority,
            "resolver_group": resolver_group,
            "sla_hours": SLA_HOURS[priority],
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate_dataset(3000)
    df.to_csv("data/synthetic_tickets.csv", index=False)
    print(f"Wrote {len(df)} synthetic tickets to data/synthetic_tickets.csv")
    print(df["category"].value_counts())
    print(df["priority"].value_counts())
