"""Central config. Every script imports paths and constants from here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "IEEE CIS Fraud Detection"
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

TRAIN_TRANSACTION = DATA_DIR / "train_transaction.csv"
TRAIN_IDENTITY = DATA_DIR / "train_identity.csv"

# Kaggle's UNLABELED competition test set. Per the hard rules: replay stream
# for the live demo only, never used for any metric.
DEMO_TRANSACTION = DATA_DIR / "test_transaction.csv"
DEMO_IDENTITY = DATA_DIR / "test_identity.csv"

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

# --- Layer 2: sliding-window spike detector --------------------------------
# Fires when the same entity (card1) produces too many risky transactions too
# fast -- a single big score is easy to hide, a burst pattern is not.
SPIKE_GROUP_COL = "card1"
SPIKE_WINDOW_SECONDS = 600      # 10-minute sliding window
SPIKE_RISKY_THRESHOLD = 0.40    # a transaction counts as "risky" for spike
                                 # purposes at the MANUAL_REVIEW floor, not BLOCK
SPIKE_COUNT_TRIGGER = 5         # >=5 risky transactions from one card1 in the
                                 # window fires an aggregate alert

# --- Layer 3: Flask API + SQLite audit trail --------------------------------
DB_PATH = ARTIFACTS / "audit_trail.db"
API_HOST = "127.0.0.1"
API_PORT = 5000
