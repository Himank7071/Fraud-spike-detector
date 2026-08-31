"""Central config. Every script imports paths and constants from here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "IEEE CIS Fraud Detection"
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

TRAIN_TRANSACTION = DATA_DIR / "train_transaction.csv"
TRAIN_IDENTITY = DATA_DIR / "train_identity.csv"

# Parquet outputs from prep_data.py
TRAIN_PARQUET = ARTIFACTS / "train.parquet"
TEST_PARQUET = ARTIFACTS / "test.parquet"

TARGET = "isFraud"
TIME_COL = "TransactionDT"
ID_COL = "TransactionID"

# Time-based split: first 70% of the timeline trains, last 30% is held out.
TRAIN_FRACTION = 0.70
SEED = 42

# Decision thresholds (Section 10 of the context doc)
BLOCK_THRESHOLD = 0.70
REVIEW_LOW = 0.40
REVIEW_HIGH = 0.60

# Business assumption for false-positive cost, in rupees
AVG_ORDER_VALUE_INR = 1200
