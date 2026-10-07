"""Synthetic salary dataset: generation, loading and train/test split.

Salary is deliberately NON-linear (experience bonus after 10 years and a
job-title x experience interaction) so that ensemble methods have real
structure to learn beyond a plain linear model.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (CATEGORICAL_FEATURES, DATA_PATH, N_SAMPLES, NUMERIC_FEATURES,
                     SEED, TARGET, TEST_SIZE)

JOB_BASE = {"Analyst": 35000, "Developer": 50000, "Data Scientist": 65000,
            "Manager": 75000, "Director": 100000}
JOB_GROWTH = {"Analyst": 1500, "Developer": 2500, "Data Scientist": 3500,
              "Manager": 3000, "Director": 5000}   # salary gain per year of experience
EDU_BONUS = {"Bachelors": 0, "Masters": 8000, "PhD": 16000}
LOC_FACTOR = {"Tier-1 City": 1.15, "Tier-2 City": 1.0, "Tier-3 City": 0.88}


def generate_dataset(n: int = N_SAMPLES, seed: int = SEED) -> pd.DataFrame:
    """Create a synthetic employee salary dataset."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "employee_id": np.arange(1, n + 1),
        "years_experience": rng.integers(0, 21, n),
        "education_level": rng.choice(list(EDU_BONUS), n, p=[0.5, 0.35, 0.15]),
        "job_title": rng.choice(list(JOB_BASE), n),
        "gender": rng.choice(["Male", "Female"], n),
        "location": rng.choice(list(LOC_FACTOR), n),
    })
    df["age"] = (22 + df.years_experience + rng.integers(0, 6, n)).clip(22, 60)

    exp = df.years_experience
    base = (df.job_title.map(JOB_BASE) + df.education_level.map(EDU_BONUS)
            + exp * df.job_title.map(JOB_GROWTH)          # job x experience interaction
            + np.clip(exp - 10, 0, None) * 2000)          # seniority bonus (non-linear)
    df[TARGET] = (base * df.location.map(LOC_FACTOR)
                  + rng.normal(0, 5000, n)).round(-2)
    return df


def load_data(path=DATA_PATH) -> pd.DataFrame:
    """Load the dataset, generating and saving it first if it does not exist."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        generate_dataset().to_csv(path, index=False)
    return pd.read_csv(path)


def split_data(df: pd.DataFrame):
    """Return X_train, X_test, y_train, y_test (80/20 split)."""
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)
