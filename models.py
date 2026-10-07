"""Model zoo: two baselines plus the four ensemble families."""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (AdaBoostRegressor, BaggingRegressor,
                              GradientBoostingRegressor, RandomForestRegressor,
                              StackingRegressor, VotingRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

from config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, SEED


def build_preprocessor() -> ColumnTransformer:
    """One-hot encode text columns, pass numeric columns through."""
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ("num", "passthrough", NUMERIC_FEATURES),
    ])


def get_models(seed: int = SEED) -> dict:
    """Return {model name: (family, estimator)}."""
    tree = lambda: DecisionTreeRegressor(max_depth=6, random_state=seed)
    rf = lambda: RandomForestRegressor(n_estimators=200, random_state=seed)
    gb = lambda: GradientBoostingRegressor(n_estimators=200, learning_rate=0.1,
                                           max_depth=3, random_state=seed)
    return {
        "Linear Regression": ("Baseline", LinearRegression()),
        "Decision Tree": ("Baseline", tree()),
        # Bagging: many models on bootstrap samples, predictions averaged
        "Bagging (Trees)": ("Bagging", BaggingRegressor(tree(), n_estimators=100,
                                                        random_state=seed)),
        "Random Forest": ("Bagging", rf()),
        # Boosting: models trained sequentially, each fixing the previous errors
        "AdaBoost": ("Boosting", AdaBoostRegressor(tree(), n_estimators=100,
                                                   random_state=seed)),
        "Gradient Boosting": ("Boosting", gb()),
        # Voting: different model types, predictions averaged
        "Voting": ("Voting", VotingRegressor([("lr", LinearRegression()),
                                              ("rf", rf()), ("gb", gb())])),
        # Stacking: a meta-model learns how to combine the base models
        "Stacking": ("Stacking", StackingRegressor(
            estimators=[("dt", tree()), ("rf", rf()), ("gb", gb())],
            final_estimator=LinearRegression(), cv=5)),
    }


def make_pipeline(estimator) -> Pipeline:
    """Preprocessing + model in one object, so raw rows can be predicted directly."""
    return Pipeline([("prep", build_preprocessor()), ("model", estimator)])
