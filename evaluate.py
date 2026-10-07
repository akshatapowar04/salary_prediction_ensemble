"""Training, evaluation and plotting helpers."""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

from config import CV_FOLDS
from models import make_pipeline

COLORS = {"Baseline": "#7f7f7f", "Bagging": "#1f77b4", "Boosting": "#d62728",
          "Voting": "#2ca02c", "Stacking": "#9467bd"}


def evaluate_models(models, X_train, X_test, y_train, y_test):
    """Fit every model; return (metrics table, dict of fitted pipelines)."""
    rows, fitted = [], {}
    for name, (family, est) in models.items():
        pipe = make_pipeline(est)
        start = time.perf_counter()
        pipe.fit(X_train, y_train)
        fit_time = time.perf_counter() - start
        pred = pipe.predict(X_test)
        cv = cross_val_score(make_pipeline(est), X_train, y_train,
                             cv=CV_FOLDS, scoring="r2").mean()
        rows.append({"Family": family, "Model": name,
                     "R2": r2_score(y_test, pred),
                     "MAE": mean_absolute_error(y_test, pred),
                     "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
                     f"CV_R2_{CV_FOLDS}fold": cv, "Fit_s": fit_time})
        fitted[name] = pipe
        print(f"  done: {name}")
    table = pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)
    return table.round({"R2": 4, "MAE": 1, "RMSE": 1, f"CV_R2_{CV_FOLDS}fold": 4, "Fit_s": 2}), fitted


def plot_comparison(table, path):
    """Horizontal bar chart of test R2, coloured by ensemble family."""
    t = table.sort_values("R2")
    plt.figure(figsize=(8, 4.5))
    plt.barh(t["Model"], t["R2"], color=[COLORS[f] for f in t["Family"]])
    plt.xlim(max(0, t["R2"].min() - 0.05), 1.0)
    plt.xlabel("R² on test set")
    plt.title("Model comparison - Salary Prediction")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS.values()]
    plt.legend(handles, COLORS.keys(), loc="lower right")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_actual_vs_pred(pipe, name, X_test, y_test, path):
    """Scatter of actual vs predicted salary for one model."""
    pred = pipe.predict(X_test)
    plt.figure(figsize=(5.5, 5.5))
    plt.scatter(y_test, pred, s=14, alpha=0.6)
    lim = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
    plt.plot(lim, lim, "k--", lw=1)
    plt.xlabel("Actual salary"); plt.ylabel("Predicted salary")
    plt.title(f"Actual vs predicted - {name}")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
