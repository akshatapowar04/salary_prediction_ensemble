"""Central configuration for the Salary Prediction Ensemble project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "salary.csv"
RESULTS_DIR = ROOT
MODEL_PATH = ROOT / "best_model.joblib"

SEED = 42
N_SAMPLES = 800
TEST_SIZE = 0.2
CV_FOLDS = 5

TARGET = "salary"
NUMERIC_FEATURES = ["years_experience", "age"]
CATEGORICAL_FEATURES = ["education_level", "job_title", "gender", "location"]
