from pathlib import Path

# --- Directory Structure ---
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_RAW_DIR = BASE_DIR / 'data'
TRAIN_PATH = DATA_RAW_DIR / 'train.csv'
TEST_PATH = DATA_RAW_DIR / 'test.csv'

MODEL_DIR = BASE_DIR / 'models'
MODEL_DIR.mkdir(exist_ok=True)

SUBMISSION_DIR = BASE_DIR / 'submissions'
SUBMISSION_DIR.mkdir(exist_ok=True)
SUBMISSION_PATH = SUBMISSION_DIR / "submission.csv"

# --- Target Information ---
# Dual-model system
TARGET_TOTAL = 'count'
TARGET_CASUAL = 'casual'
TARGET_REGISTERED = 'registered'

# --- Feature Groups ---
# Defining these here makes your ColumnTransformer much cleaner
FEATURES_NUMERICAL = ['temp', 'atemp', 'humidity', 'windspeed']
FEATURES_CATEGORICAL = ['season', 'holiday', 'workingday', 'weather']
FEATURES_TEMPORAL = 'datetime'  # Single column to be processed by our custom extractor

# --- Prophet features ---
PROPHET_REGRESSORS = ['temp', 'humidity', 'workingday', 'weather', 'hour']

# --- Time-based Specialization ---
# Nigh: 0h - 5H
# Day: 6h - 23h
NIGHT_HOURS = [0, 1, 2, 3, 4, 5]

# --- Interaction Terms ---
# Define which pairs to interact to avoid feature explosion
INTERACTION_PAIRS = [
    ('hour', 'workingday'),
    ('hour', 'season'),
    ('temp', 'humidity')
]