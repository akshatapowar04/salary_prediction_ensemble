"""Run the full ensemble comparison:  python main.py"""
import joblib

from config import MODEL_PATH, RESULTS_DIR
from data import load_data, split_data
from evaluate import evaluate_models, plot_actual_vs_pred, plot_comparison
from models import get_models


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    df = load_data()
    print(f"Dataset: {df.shape[0]} rows, {df.isnull().sum().sum()} missing values")
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"Train: {len(X_train)}  Test: {len(X_test)}\nTraining models...")

    table, fitted = evaluate_models(get_models(), X_train, X_test, y_train, y_test)
    print("\n" + table.to_string(index=False))

    best = table.loc[0, "Model"]
    print(f"\nBest model: {best} (R2 = {table.loc[0, 'R2']})")

    table.to_csv(RESULTS_DIR / "metrics.csv", index=False)
    plot_comparison(table, RESULTS_DIR / "model_comparison.png")
    plot_actual_vs_pred(fitted[best], best, X_test, y_test,
                        RESULTS_DIR / "best_actual_vs_predicted.png")
    joblib.dump(fitted[best], MODEL_PATH)
    print(f"Saved metrics, plots and best model to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
