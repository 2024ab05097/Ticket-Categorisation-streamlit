"""
Global configuration
Used throughout the project.

Changing values here automatically updates the
entire project.
"""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

DATA_DIR = ROOT_DIR / "data"

ARTIFACT_DIR = ROOT_DIR / "artifacts"

MODEL_DIR = ROOT_DIR / "models"

LOG_DIR = ROOT_DIR / "logs"

OUTPUT_DIR = ROOT_DIR / "outputs"

#########################################
# Model Selection
#########################################

EMBEDDING_BACKEND = "transformer"
# tfidf
# transformer

MODEL_TYPE = "logreg"
# logreg
# mlp

#########################################
# Confidence
#########################################

CONFIDENCE_THRESHOLD = 0.80

LLM_TRIGGER_THRESHOLD = 0.60

#########################################
# Active Learning
#########################################

ACTIVE_LEARNING_BATCH_SIZE = 100

AUTO_RETRAIN_THRESHOLD = 250

#########################################
# Training
#########################################

TEST_SIZE = 0.20

VALIDATION_SIZE = 0.10

RANDOM_STATE = 42

#########################################
# BERT
#########################################

TRANSFORMER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

#########################################
# Logging
#########################################

LOG_LEVEL = "INFO"

#########################################
# MLflow
#########################################

MLFLOW_EXPERIMENT = "Ticket_Triage_Dissertation"

#########################################
# Streamlit
#########################################

PAGE_TITLE = "Agentic Ticket Triaging"

PAGE_ICON = "🤖"

#########################################
# SLA
#########################################

SLA_HOURS = {

    "P1-Critical":2,

    "P2-High":8,

    "P3-Medium":24,

    "P4-Low":72

}
