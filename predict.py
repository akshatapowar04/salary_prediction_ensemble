"""Predict a salary with the saved best model.

Example:
  python predict.py --experience 6 --education Masters --job "Data Scientist" \
                    --age 30 --gender Female --location "Tier-1 City"
"""
import argparse

import joblib
import pandas as pd

from config import MODEL_PATH


def main():
    p = argparse.ArgumentParser(description="Predict annual salary")
    p.add_argument("--experience", type=int, required=True)
    p.add_argument("--education", choices=["Bachelors", "Masters", "PhD"], required=True)
    p.add_argument("--job", required=True,
                   choices=["Analyst", "Developer", "Data Scientist", "Manager", "Director"])
    p.add_argument("--age", type=int, required=True)
    p.add_argument("--gender", choices=["Male", "Female"], required=True)
    p.add_argument("--location", choices=["Tier-1 City", "Tier-2 City", "Tier-3 City"],
                   required=True)
    a = p.parse_args()

    if not MODEL_PATH.exists():
        raise SystemExit("No saved model found - run `python main.py` first.")
    row = pd.DataFrame([{"years_experience": a.experience, "age": a.age,
                         "education_level": a.education, "job_title": a.job,
                         "gender": a.gender, "location": a.location}])
    salary = joblib.load(MODEL_PATH).predict(row)[0]
    print(f"Predicted annual salary: {round(salary, -2):,.0f}")


if __name__ == "__main__":
    main()
